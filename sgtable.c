/*
 * sgtable.c - Recover a process's dma-buf scatter-gather page lists (LEMON add-on).
 *
 * Motivation (Tensor/Pixel): the EdgeTPU KV cache is in samsung_dma_heap "system" dma-bufs whose
 * pages are IOMMU-scattered and whose CPU mapping reads zero. To carve the KV from a physical dump
 * we need each buffer's exact physical page list. This walks kernel structures ENTIRELY from
 * userspace using LEMON's existing read primitive (read_kernel_memory) as a pointer-deref oracle:
 *
 *   init_task --(task list)--> task_struct(tgid==pid)
 *     -> files_struct -> fdtable -> fd[] -> struct file
 *        -> (f_op == dma_buf_fops) -> struct dma_buf { size, exp_name, priv }
 *           -> samsung_dma_buffer.priv --(heuristic)--> struct sg_table
 *              -> scatterlist walk -> (struct page*, offset, length)
 *
 * Struct offsets come from vmlinux BTF (libbpf) at runtime; `samsung_dma_buffer` is not in BTF so
 * its embedded sg_table is located heuristically. Output is CSV on stdout:
 *     buf,<fd>,<size>,<exp_name>
 *     seg,<fd>,<page_hex>,<offset>,<length>
 * feed to testexp/sgtable/reassemble_kv.py to carve + reassemble from the LEMON image.
 *
 * Enabled via `-S/--sgtable PID`. Requires kptr_restrict=0 (init_task/dma_buf_fops from kallsyms).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <bpf/btf.h>

#include "lemon.h"

extern int read_kernel_memory(const uintptr_t addr, const size_t size, const unsigned char **restrict data);

/* ---- read helpers: read_kernel_memory shares one buffer, so copy out immediately ---- */
static int rd(uintptr_t va, void *out, size_t n) {
    const unsigned char *d = NULL;
    if (!va) return -1;
    int ret = read_kernel_memory(va, n, &d);
    if (ret < 0 || !d) return -1;
    memcpy(out, d, n);
    return 0;
}
static uint64_t rd_u64(uintptr_t va) { uint64_t v = 0; return rd(va, &v, 8) ? 0 : v; }
static uint32_t rd_u32(uintptr_t va) { uint32_t v = 0; return rd(va, &v, 4) ? 0 : v; }
static void rd_str(uintptr_t va, char *out, size_t n) {
    out[0] = 0;
    if (!va) return;
    const unsigned char *d = NULL;
    if (read_kernel_memory(va, n - 1, &d) < 0 || !d) return;
    memcpy(out, d, n - 1); out[n - 1] = 0;
    for (size_t i = 0; i < n - 1; i++) if (!out[i]) break;
}

/* ---- BTF: member byte offset / struct size (robust vs hardcoding) ---- */
static struct btf *g_btf = NULL;
static long moff(const char *type, const char *member) {
    int tid = btf__find_by_name_kind(g_btf, type, BTF_KIND_STRUCT);
    if (tid < 0) { ERR("sgtable: struct %s not in BTF", type); return -1; }
    const struct btf_type *t = btf__type_by_id(g_btf, tid);
    const struct btf_member *m = btf_members(t);
    for (__u16 i = 0; i < btf_vlen(t); i++, m++) {
        const char *nm = btf__name_by_offset(g_btf, m->name_off);
        if (nm && !strcmp(nm, member)) return btf_member_bit_offset(t, i) / 8;
    }
    ERR("sgtable: %s.%s not found in BTF", type, member);
    return -1;
}
static long ssize_of(const char *type) {
    int tid = btf__find_by_name_kind(g_btf, type, BTF_KIND_STRUCT);
    if (tid < 0) return -1;
    return btf__resolve_size(g_btf, tid);
}

/* ---- kallsyms lookup (kptr_restrict=0 required; LEMON already checks CAP_SYSLOG) ---- */
static uintptr_t ksym(const char *name) {
    FILE *fp = fopen("/proc/kallsyms", "r");
    if (!fp) { ERR("sgtable: cannot open /proc/kallsyms"); return 0; }
    char line[512], nm[256]; uintptr_t addr = 0, found = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "%" SCNxPTR " %*c %255s", &addr, nm) == 2 && !strcmp(nm, name)) {
            found = addr; break;
        }
    }
    fclose(fp);
    if (!found) ERR("sgtable: symbol %s not found (kptr_restrict / module?)", name);
    return found;
}

#define SG_CHAIN 0x01UL
#define SG_END   0x02UL
#define SG_MASK  0x03UL
#define KVA_MIN  0xffff000000000000ULL

/* Locate the embedded struct sg_table inside samsung_dma_buffer priv (not in BTF).
 * sg_table = { struct scatterlist *sgl; unsigned int nents; unsigned int orig_nents; }. */
static int find_sgtable(uintptr_t priv, uintptr_t *sgl_out) {
    for (int off = 0; off <= 0x100; off += 8) {
        uint64_t sgl = rd_u64(priv + off);
        uint32_t nents = rd_u32(priv + off + 8);
        uint32_t orig  = rd_u32(priv + off + 12);
        if (sgl < KVA_MIN || orig == 0 || orig >= 100000) continue;
        if (nents != orig && nents != 0) continue;
        long l_off = moff("scatterlist", "length");
        if (l_off < 0) return -1;
        uint32_t first_len = rd_u32(sgl + l_off);
        if (first_len == 0) continue;
        *sgl_out = sgl;
        return 0;
    }
    return -1;
}

int run_sgtable(struct lemon_ctx *restrict ctx, pid_t target) {
    g_btf = btf__load_vmlinux_btf();
    if (!g_btf) { ERR("sgtable: btf__load_vmlinux_btf failed"); return EINVAL; }

    uintptr_t init_task = ksym("init_task");
    uintptr_t dma_buf_fops = ksym("dma_buf_fops");
    if (!init_task || !dma_buf_fops) return ENOENT;

    long o_tasks = moff("task_struct", "tasks");
    long o_tgid  = moff("task_struct", "tgid");
    long o_files = moff("task_struct", "files");
    long o_fdt   = moff("files_struct", "fdt");
    long o_maxfd = moff("fdtable", "max_fds");
    long o_fd    = moff("fdtable", "fd");
    long o_fop   = moff("file", "f_op");
    long o_priv  = moff("file", "private_data");
    long o_size  = moff("dma_buf", "size");
    long o_exp   = moff("dma_buf", "exp_name");
    long o_dpriv = moff("dma_buf", "priv");
    long o_pl    = moff("scatterlist", "page_link");
    long o_soff  = moff("scatterlist", "offset");
    long o_slen  = moff("scatterlist", "length");
    long sg_sz   = ssize_of("scatterlist");
    if ((o_tasks|o_tgid|o_files|o_fdt|o_maxfd|o_fd|o_fop|o_priv|o_size|o_exp|o_dpriv|
         o_pl|o_soff|o_slen) < 0 || sg_sz <= 0) {
        ERR("sgtable: missing BTF offsets"); return EINVAL;
    }

    /* Walk the task list from init_task to find the target tgid. */
    uintptr_t task = init_task, found = 0;
    for (int guard = 0; guard < 100000; guard++) {
        uintptr_t next_link = rd_u64(task + o_tasks);      /* &next->tasks */
        if (!next_link) break;
        uintptr_t next = next_link - o_tasks;
        if (next == init_task) break;
        if ((uint32_t)rd_u32(next + o_tgid) == (uint32_t)target) { found = next; break; }
        task = next;
    }
    if (!found) { ERR("sgtable: pid %d not found in task list", target); return ESRCH; }
    INFO("sgtable: task_struct for pid %d @ 0x%" PRIxPTR, target, found);

    uintptr_t files = rd_u64(found + o_files);
    uintptr_t fdt   = files ? rd_u64(files + o_fdt) : 0;
    if (!fdt) { ERR("sgtable: no fdtable"); return EINVAL; }
    uint32_t max_fds = rd_u32(fdt + o_maxfd);
    uintptr_t fdarr  = rd_u64(fdt + o_fd);
    if (!fdarr || max_fds == 0 || max_fds > 1 << 20) { ERR("sgtable: bad fdtable"); return EINVAL; }

    int nbuf = 0, nseg = 0;
    for (uint32_t fd = 0; fd < max_fds; fd++) {
        uintptr_t file = rd_u64(fdarr + (uint64_t)fd * 8);
        if (!file) continue;
        if (rd_u64(file + o_fop) != dma_buf_fops) continue;
        uintptr_t db = rd_u64(file + o_priv);
        if (!db) continue;
        uint64_t size = rd_u64(db + o_size);
        if (size < ctx->opts.sgtable_min || size > ctx->opts.sgtable_max) continue;
        char exp[24]; rd_str(rd_u64(db + o_exp), exp, sizeof(exp));
        uintptr_t priv = rd_u64(db + o_dpriv);
        uintptr_t sgl = 0;
        if (!priv || find_sgtable(priv, &sgl) < 0) {
            DBG("sgtable: fd=%u size=%" PRIu64 " (%s): sg_table not located", fd, size, exp);
            continue;
        }
        printf("buf,%u,%" PRIu64 ",%s\n", fd, size, exp);
        nbuf++;
        uintptr_t sg = sgl;
        for (int i = 0; i < 4096; i++) {
            uint64_t pl = rd_u64(sg + o_pl);
            if (pl & SG_CHAIN) { sg = (uintptr_t)(pl & ~SG_MASK); continue; }
            uint32_t off = rd_u32(sg + o_soff);
            uint32_t len = rd_u32(sg + o_slen);
            printf("seg,%u,0x%" PRIx64 ",%u,%u\n", fd, (uint64_t)(pl & ~SG_MASK), off, len);
            nseg++;
            if (pl & SG_END) break;
            sg += sg_sz;
        }
    }
    fflush(stdout);
    INFO("sgtable: %d dma-buf(s), %d segment(s) emitted for pid %d", nbuf, nseg, target);
    btf__free(g_btf);
    return nbuf ? 0 : ENOENT;
}
