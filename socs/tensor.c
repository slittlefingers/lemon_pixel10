#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <unistd.h>

#include "../lemon.h"

/* LEMON's kernel-memory read oracle (defined in mem.c); used to read kimage_voffset's value. */
extern int read_kernel_memory(const uintptr_t addr, const size_t size, const unsigned char **restrict data);
/* Physical->kernel-virtual (linear map) translation, defined in mem.c; used to read PT pages by PA. */
extern uintptr_t phys_to_virt(const struct lemon_ctx *restrict ctx, uintptr_t phy_addr);

/*
 * tensor.c - Google Tensor (Pixel) support for LEMON.
 *
 * Problem (see tensor-recon/docs/recon-findings.md section 9):
 *   Tensor SoCs enforce S2MPU protection over firmware carveouts (tpu_fw, gpu_fw,
 *   gxp_fw, gsa, aoc, ...). A CPU read that touches such a region (or the System-RAM
 *   page immediately adjacent to it) raises an *asynchronous SError* that the kernel
 *   cannot fix up -> hard reboot. This differs from pKVM stage-2 protection, which
 *   raises a *synchronous* abort recoverable as -EFAULT (LEMON already handles that).
 *
 * Strategy (analogous to socs/qcom.c, but the danger list is queried, not magic-marked):
 *   qcom  : per-page software marker (page->private == 0xEEEEEEEE)         -> dynamic
 *   tensor: device-tree reserved-memory "no-map" carveouts                 -> queried at runtime
 *
 *   We DO NOT hardcode addresses: every Pixel model / firmware build can differ.
 *   At init we parse /sys/firmware/devicetree/base/reserved-memory and record every
 *   no-map carveout, expanded by a safety margin to cover S2MPU over-protection of the
 *   adjacent System-RAM page. dump.c then skips+pattern-fills any page that lands in a
 *   recorded range, exactly like the Qualcomm secure-page path.
 */

/* Max carveouts we track: device-tree no-map nodes + /proc/iomem reserved ranges. */
#define TENSOR_MAX_CARVEOUTS 512

/*
 * Safety margins applied on both sides of a carveout to cover S2MPU spilling slightly past the
 * device-tree boundary (or a CPU prefetch crossing into the carveout).
 *
 * We now use TWO margins instead of one blanket 2 MB:
 *   - FW_MARGIN (firmware no-map carveouts): stay generous. Firmware carveouts have NO kernel data
 *     adjacent, so a wide margin costs nothing and keeps us safe near the real S2MPU regions.
 *   - RESV_MARGIN (/proc/iomem "reserved"): tiny. These reserved ranges sit next to live kernel
 *     slab / kernel-image memory (init_task, swapper_pg_dir, sg_tables, dma_buf structs, name
 *     strings). A 2 MB margin here was pure collateral — it pattern-filled readable metadata. A
 *     small margin still covers prefetch/S2MPU-spill while returning that metadata to the dump.
 * The kernel-image override in tensor_is_protected_page() additionally forces the kernel image to
 * be read even if it lands inside a margin (the CPU executes from it, so it is never protected).
 * TUNE via docs/test-runbook.md if a boundary page ever faults.
 */
#define TENSOR_FW_MARGIN   0x200000ULL   /* 2 MB around firmware no-map carveouts */
#define TENSOR_RESV_MARGIN 0x10000ULL    /* 64 KB around /proc/iomem reserved ranges */

struct tensor_range {
    uint64_t start;    /* inclusive */
    uint64_t end;      /* inclusive */
    char     name[64]; /* carveout node name, for debug */
};

static struct tensor_range g_ranges[TENSOR_MAX_CARVEOUTS];
static int g_nranges = 0;

/* Candidate device-tree mount points, in preference order. */
static const char *DT_BASES[] = {
    "/sys/firmware/devicetree/base/reserved-memory",
    "/proc/device-tree/reserved-memory",
};

/* Read a big-endian unsigned 64-bit value from an 8-byte buffer. */
static uint64_t be64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | p[i];
    return v;
}

/* True if a sibling property file exists in this node directory (e.g. "no-map"). */
static bool node_has(const char *node_dir, const char *prop) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", node_dir, prop);
    return access(path, F_OK) == 0;
}

/*
 * node_disabled() - True if the reserved-memory node's "status" property reads "disabled".
 *
 * A device-tree reserved-memory node with status="disabled" is NEVER claimed by the kernel's
 * reserved-memory framework: it does not appear in reserved_mem[], is not carved out of System RAM,
 * and is not S2MPU-protected. Its range is ordinary readable RAM (confirmed: s2m@c0000000, a
 * disabled 74MB "no-map" node, reads fine via a physical -r read and holds live struct dma_buf
 * metadata). Avoiding it purely because it carries "no-map" pattern-fills readable memory.
 * A missing status property means the node is enabled by default (return false → keep avoiding).
 */
static bool node_disabled(const char *node_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/status", node_dir);
    FILE *fp = fopen(path, "r");
    if (!fp) return false;                       /* no status -> enabled by default */
    char buf[16] = {0};
    fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    return strncmp(buf, "disabled", 8) == 0;     /* only skip an explicitly disabled node */
}

/*
 * add_range() - record one [base, base+size) carveout, expanded by the caller's margin.
 * Stored inclusive [start, end]. Margin is clamped at 0 on the low side.
 */
static void add_range(const struct lemon_ctx *restrict ctx, const char *name, uint64_t base, uint64_t size, uint64_t margin) {
    if (size == 0 || g_nranges >= TENSOR_MAX_CARVEOUTS) return;

    uint64_t lo = (base > margin) ? base - margin : 0;
    uint64_t hi = base + size - 1 + margin;

    g_ranges[g_nranges].start = lo;
    g_ranges[g_nranges].end   = hi;
    snprintf(g_ranges[g_nranges].name, sizeof(g_ranges[g_nranges].name), "%s", name);
    g_nranges++;

    DBG("tensor: avoid %-28s phys 0x%llx-0x%llx (carveout 0x%llx+0x%llx +/-margin)",
        name, (unsigned long long)lo, (unsigned long long)hi,
        (unsigned long long)base, (unsigned long long)size);
}

/*
 * CMA handling - bitmap-aware. A CMA area (video vframe/vstream, dma pools, pkvm_iommu_cma) is a MIX:
 *   - pages cma_alloc'd to a device (video codec, pKVM IOMMU) are S2MPU-protected. A CPU read raises
 *     a FATAL async SError -> reboot.
 *   - pages the buddy allocator lends to MOVABLE allocations (anon, zsmalloc swap) are ordinary
 *     readable RAM. The runtime name->buffer map's swapped pages land here.
 * A whole-region whitelist is UNSAFE (it reads the device pages and reboots). Instead we read each
 * region's struct cma.bitmap and decide PER PAGE: bit SET = cma_alloc'd (device, protected -> fill),
 * bit CLEAR = free/movable (readable -> read). CMA placement + bitmap are dynamic per boot, so we
 * read them live each capture from the kernel's cma_areas[] (never hardcode).
 *
 * struct cma (units: pages): base_pfn@0, count@8, bitmap@16 (ulong*), order_per_bit@24. The array
 * stride = sizeof(struct cma) is config-dependent (spinlock_t grows under lockdep). Verify with:
 *     pahole -C cma vmlinux        (or from the vmlinux BTF)
 * A wrong stride / bad entry is dropped by the sanity check, falling back to conservative avoidance.
 *
 * WARNING: async SError is unforgiving - one protected page read = reboot. This relies on
 * "S2MPU-protected set == cma-bitmap SET set" (validated offline on this SoC: map pages all CLEAR,
 * pkvm_iommu_cma all SET). A snapshot race (a device allocating between init and the page read) is
 * inherent - dump with the target app frozen and the system as quiescent as possible.
 */
#ifndef TENSOR_CMA_STRIDE
#define TENSOR_CMA_STRIDE 136   /* sizeof(struct cma) on 6.6 android15-4k; set from BTF/pahole per build */
#endif
#define TENSOR_MAX_CMA 32
enum { CMA_NOT = 0, CMA_READABLE = 1, CMA_DEVICE = 2 };

struct tensor_cma {
    uint64_t base_pfn;
    uint64_t npages;         /* struct cma.count */
    uint32_t order_per_bit;
    uint64_t nbits;          /* npages >> order_per_bit */
    uint8_t *bitmap;         /* malloc'd snapshot; bit SET = device (protected), CLEAR = readable */
};
static struct tensor_cma g_cma[TENSOR_MAX_CMA];
static int g_ncma = 0;

/* True if pfn's bit is set (device-allocated -> protected). Unknown -> treat as protected (safe). */
static bool cma_bit_set(const struct tensor_cma *c, uint64_t pfn) {
    uint64_t idx = (pfn - c->base_pfn) >> c->order_per_bit;
    if (idx >= c->nbits || !c->bitmap) return true;
    return (c->bitmap[idx >> 3] >> (idx & 7)) & 1;
}

/* Classify a physical address against the CMA bitmaps. */
static int cma_status(uint64_t pa) {
    uint64_t pfn = pa >> 12;
    for (int i = 0; i < g_ncma; i++)
        if (pfn >= g_cma[i].base_pfn && pfn < g_cma[i].base_pfn + g_cma[i].npages)
            return cma_bit_set(&g_cma[i], pfn) ? CMA_DEVICE : CMA_READABLE;
    return CMA_NOT;
}

/* True if a physical address lies in any CMA region (regardless of bitmap). */
static bool cma_contains(uint64_t pa) {
    uint64_t pfn = pa >> 12;
    for (int i = 0; i < g_ncma; i++)
        if (pfn >= g_cma[i].base_pfn && pfn < g_cma[i].base_pfn + g_cma[i].npages) return true;
    return false;
}

/* True if inclusive [start,end] touches a CMA region containing at least one device (protected)
 * page - such a huge chunk MUST be reprocessed at PAGE_SIZE so the per-page bitmap applies. */
static bool cma_range_has_device(uint64_t start, uint64_t end) {
    for (int i = 0; i < g_ncma; i++) {
        uint64_t rs = g_cma[i].base_pfn << 12;
        uint64_t re = ((g_cma[i].base_pfn + g_cma[i].npages) << 12) - 1;
        if (start > re || end < rs) continue;
        uint64_t p0 = (start > rs ? start : rs) >> 12;
        uint64_t p1 = (end   < re ? end   : re) >> 12;
        for (uint64_t pfn = p0; pfn <= p1; pfn++)
            if (cma_bit_set(&g_cma[i], pfn)) return true;
    }
    return false;
}

/*
 * tensor_init_cma() - read cma_areas[] AND each region's allocation bitmap from the kernel.
 * Uses the same /proc/kallsyms + read_kernel_memory oracle as tensor_init_kimage(). Dynamic per boot.
 */
static void tensor_init_cma(const struct lemon_ctx *restrict ctx) {
    uintptr_t areas = 0, count_addr = 0;
    unsigned long addr;
    char line[512], sym[128];
    FILE *fp = fopen("/proc/kallsyms", "r");
    if (!fp) return;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "%lx %*c %127s", &addr, sym) != 2) continue;
        if      (!strcmp(sym, "cma_areas"))      areas = (uintptr_t)addr;
        else if (!strcmp(sym, "cma_area_count")) count_addr = (uintptr_t)addr;
    }
    fclose(fp);
    if (!areas || !count_addr) { DBG("tensor: cma_areas/cma_area_count not in kallsyms; CMA not handled"); return; }

    const unsigned char *d = NULL;
    if (read_kernel_memory(count_addr, 4, &d) < 0 || !d) return;
    unsigned int n = *(const unsigned int *)d;
    if (n > TENSOR_MAX_CMA) n = TENSOR_MAX_CMA;

    for (unsigned int i = 0; i < n; i++) {
        const unsigned char *e = NULL;
        if (read_kernel_memory(areas + (uintptr_t)i * TENSOR_CMA_STRIDE, 32, &e) < 0 || !e) continue;
        uint64_t base_pfn = *(const uint64_t *)e;
        uint64_t cnt      = *(const uint64_t *)(e + 8);
        uint64_t bmp_ptr  = *(const uint64_t *)(e + 16);
        uint32_t opb      = *(const uint32_t *)(e + 24);
        /* Sanity-guard a wrong stride / empty slot: plausible pfn, page count and bitmap pointer. */
        if (base_pfn == 0 || cnt == 0 || base_pfn > 0x2000000ULL || cnt > 0x1000000ULL || !bmp_ptr) continue;
        if (g_ncma >= TENSOR_MAX_CMA) break;

        uint64_t nbits  = cnt >> opb;
        size_t   nbytes = (size_t)((nbits + 7) / 8);
        uint8_t *bm = calloc(1, nbytes + 8);
        if (!bm) continue;
        /* Read the bitmap 8 bytes at a time (the proven read_kernel_memory granule). */
        int ok = 1;
        for (size_t off = 0; off < nbytes; off += 8) {
            const unsigned char *w = NULL;
            if (read_kernel_memory((uintptr_t)bmp_ptr + off, 8, &w) < 0 || !w) { ok = 0; break; }
            size_t chunk = (nbytes - off < 8) ? (nbytes - off) : 8;
            memcpy(bm + off, w, chunk);
        }
        if (!ok) { free(bm); continue; }

        struct tensor_cma *c = &g_cma[g_ncma++];
        c->base_pfn = base_pfn; c->npages = cnt; c->order_per_bit = opb;
        c->nbits = nbits; c->bitmap = bm;
        uint64_t setc = 0;
        for (uint64_t b = 0; b < nbits; b++) if ((bm[b >> 3] >> (b & 7)) & 1) setc++;
        INFO("tensor: CMA phys 0x%llx-0x%llx (%llu MB): %llu device page(s) protected, rest readable",
             (unsigned long long)(base_pfn << 12), (unsigned long long)(((base_pfn + cnt) << 12) - 1),
             (unsigned long long)(cnt >> 8), (unsigned long long)setc);
    }
    INFO("tensor: %d CMA region(s) parsed from kernel cma_areas[] with per-page bitmap (dynamic per boot)", g_ncma);
}

/*
 * Kernel page-table "allow-set" (force-read override).
 *
 * Root cause of the recurring over-fill bugs (swapper_pg_dir, init_task, vmemmap PTs): the avoid-list
 * over-approximates "fatal to read" from DT no-map + /proc/iomem "reserved" + margins, and repeatedly
 * catches READABLE kernel memory. Kernel page-table pages are the worst victims: the MMU must walk
 * them, so they are ALWAYS CPU-readable, yet they get pattern-filled when they land in a carveout
 * margin (e.g. a vmemmap L2 table at an iomem-reserved page). Filling a PT page then breaks address
 * translation in the dump (vmemmap -> struct page, page-table walks) offline.
 *
 * Fix: at init, walk swapper_pg_dir live (via the eBPF read oracle, EL1 - no pKVM/EL2 needed),
 * collect every intermediate PT page's physical address, and force-read them in tensor_is_protected_page
 * (like the kimage override). The set is tiny (~379 pages / 1.5 MB on this SoC) and bounded.
 *
 * NOTE: this covers KERNEL page tables only (init_mm/swapper). They are stable during a capture and
 * always live inside a scanned System-RAM range, so the record-PFN + main-loop override is equivalent
 * to reading them at discovery. Per-process USER page tables (task->mm->pgd), which are volatile, are a
 * follow-up that needs read-on-discovery (or a frozen target) - not handled here.
 */
#define TENSOR_MAX_PT_PAGES 16384
#define TENSOR_PT_OA_MASK   0x0000FFFFFFFFF000ULL   /* table-descriptor output address bits [47:12] */
static uint64_t g_pt_pages[TENSOR_MAX_PT_PAGES];    /* page-aligned PAs of kernel PT pages */
static int g_npt = 0;
static int g_pt_ready = 0;

static int tensor_pt_cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Linear scan over the (unsorted, growing) set - dedup / cycle guard DURING the walk. n is small. */
static bool tensor_pt_seen(uint64_t pa) {
    for (int i = 0; i < g_npt; i++) if (g_pt_pages[i] == pa) return true;
    return false;
}

/* Binary search over the sorted set - used per dumped page, so it must be fast. */
static bool tensor_pt_is_pagetable(uint64_t pa) {
    if (!g_npt) return false;
    uint64_t key = pa & ~(uint64_t)(PAGE_SIZE - 1);
    int lo = 0, hi = g_npt - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if      (g_pt_pages[m] == key) return true;
        else if (g_pt_pages[m] <  key) lo = m + 1;
        else                           hi = m - 1;
    }
    return false;
}

/*
 * tensor_pt_walk() - recurse one page-table level, recording each table page.
 * @pa: physical address of this table page. @level: 0=PGD,1=PMD,2=PTE (VA_BITS=39, 3 levels).
 * The read oracle returns a shared buffer, so copy the page out before recursing (child reads clobber it).
 */
static void tensor_pt_walk(const struct lemon_ctx *restrict ctx, uint64_t pa, int level) {
    if (level > 2 || g_npt >= TENSOR_MAX_PT_PAGES) return;
    pa &= ~(uint64_t)(PAGE_SIZE - 1);
    if (tensor_pt_seen(pa)) return;
    const unsigned char *d = NULL;
    if (read_kernel_memory(phys_to_virt(ctx, pa), PAGE_SIZE, &d) < 0 || !d) return;
    unsigned char page[PAGE_SIZE];
    memcpy(page, d, PAGE_SIZE);
    g_pt_pages[g_npt++] = pa;
    if (level >= 2) return;                    /* PTE table: entries map data pages, not child tables */
    for (int i = 0; i < (int)(PAGE_SIZE / 8); i++) {
        uint64_t e;
        memcpy(&e, page + i * 8, 8);
        if ((e & 3) == 3)                      /* table descriptor -> next-level table */
            tensor_pt_walk(ctx, e & TENSOR_PT_OA_MASK, level + 1);
    }
}

/*
 * tensor_init_pt_allowset() - build the kernel page-table allow-set from swapper_pg_dir.
 * The root PGD is read via its kernel VA (it lives in the kernel image); its child tables are read by PA.
 */
static void tensor_init_pt_allowset(const struct lemon_ctx *restrict ctx) {
    g_pt_ready = 1;
    uintptr_t swapper = 0;
    unsigned long addr;
    char line[512], sym[128];
    FILE *fp = fopen("/proc/kallsyms", "r");
    if (!fp) return;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "%lx %*c %127s", &addr, sym) != 2) continue;
        if (!strcmp(sym, "swapper_pg_dir")) { swapper = (uintptr_t)addr; break; }
    }
    fclose(fp);
    if (!swapper) { DBG("tensor: swapper_pg_dir not in kallsyms; PT allow-set disabled"); return; }

    const unsigned char *d = NULL;
    if (read_kernel_memory(swapper, PAGE_SIZE, &d) < 0 || !d) { DBG("tensor: cannot read swapper_pg_dir"); return; }
    unsigned char root[PAGE_SIZE];
    memcpy(root, d, PAGE_SIZE);
    for (int i = 0; i < (int)(PAGE_SIZE / 8); i++) {
        uint64_t e;
        memcpy(&e, root + i * 8, 8);
        if ((e & 3) == 3)                      /* PGD table descriptor -> PMD table */
            tensor_pt_walk(ctx, e & TENSOR_PT_OA_MASK, 1);
    }
    qsort(g_pt_pages, g_npt, sizeof(uint64_t), tensor_pt_cmp);
    INFO("tensor: kernel page-table allow-set = %d PT pages force-read (unfilled even inside carveouts)", g_npt);
}

/*
 * parse_node_reg() - read a carveout node's "reg" and add its ranges.
 * @node_dir: absolute path to the reserved-memory child directory.
 *
 * reg layout on arm64 Tensor: #address-cells = #size-cells = 2, so each entry is
 * 16 bytes (8-byte big-endian base, 8-byte big-endian size). A node may carry several
 * entries; we parse all whole 16-byte records and ignore a node whose reg is absent
 * (dynamically-placed nodes) or not a multiple of 16.
 */
static void parse_node_reg(const struct lemon_ctx *restrict ctx, const char *node_dir, const char *name) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/reg", node_dir);

    FILE *fp = fopen(path, "rb");
    if (!fp) return;                  /* no fixed address (dynamic reserved-memory) */

    uint8_t buf[16 * 16];             /* up to 16 reg entries */
    size_t n = fread(buf, 1, sizeof(buf), fp);
    fclose(fp);

    if (n == 0 || n % 16 != 0) {
        DBG("tensor: %s has reg of %zu bytes (not 16-aligned), skipping", name, n);
        return;
    }
    for (size_t off = 0; off + 16 <= n; off += 16)
        add_range(ctx, name, be64(buf + off), be64(buf + off + 8), TENSOR_FW_MARGIN);
}

/*
 * add_iomem_reserved() - add "reserved" ranges nested inside System RAM (from /proc/iomem).
 *
 * Second danger source (see recon-findings.md section 10): LEMON dumps "System RAM"
 * ranges but does NOT subtract the reserved children nested within them (iomem.c
 * range_subtract() is unused). Some nested reserved regions really are S2MPU-fatal, so we
 * add every /proc/iomem "reserved" range to the avoid-list.
 *
 * EXCEPTION - CMA: large "reserved" children like 0xacc000000-0xaf83fffff (the vframe/vstream
 * video CMA pools) are NOT firmware - they are CPU-readable movable RAM the allocator uses for
 * movable/zsmalloc pages. Blanket-filling them destroyed readable data (the swapped scudo heap
 * with the runtime name->buffer map). We therefore whitelist CMA (tensor_init_cma, from the kernel's
 * cma_areas[]) and skip any reserved range that falls inside it. Must run tensor_init_cma() first.
 */
static void add_iomem_reserved(const struct lemon_ctx *restrict ctx) {
    FILE *fp = fopen("/proc/iomem", "r");
    if (!fp) { DBG("tensor: cannot open /proc/iomem for reserved scan"); return; }

    char line[512];
    int added = 0, skipped_cma = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long s, e;
        int pos = 0;
        /* "  <start>-<end> : <label>" ; %llx skips the indentation whitespace. */
        if (sscanf(line, "%llx-%llx : %n", &s, &e, &pos) < 2 || pos == 0)
            continue;
        if (strcasestr(line + pos, "reserved")) {
            /* CMA pools show as "reserved"; skip them here - decided per-page by the CMA bitmap
             * (device pages filled, movable/free pages read) in tensor_is_protected_page(). */
            if (cma_contains(s) && cma_contains(e)) { skipped_cma++; continue; }
            add_range(ctx, "iomem-reserved", s, e - s + 1, TENSOR_RESV_MARGIN);
            added++;
        }
    }
    fclose(fp);
    INFO("tensor: +%d /proc/iomem reserved ranges added to avoid-list (%d CMA ranges skipped as readable)",
         added, skipped_cma);
}

/*
 * check_init_tensor() - Detect Google Tensor and build the S2MPU carveout avoid-list.
 * @ctx: needs is_android + soc_manufacturer/soc_model from collect_android_info().
 *
 * Returns 0 if not a Tensor device (leaves ctx->is_tensor false), 1 on success,
 * or an errno-style code on failure.
 */
int check_init_tensor(struct lemon_ctx *restrict ctx) {

    /* Only Google Tensor Android devices. ro.soc.manufacturer = "Google", model "Tensor *". */
    if (!ctx->is_android) return 0;
    if (!strcasestr(ctx->soc_manufacturer, "Google") &&
        !strcasestr(ctx->soc_model, "Tensor"))
        return 0;

    ctx->is_tensor = true;
    INFO("Device uses Google Tensor SoC (S2MPU carveout avoidance enabled)");
    if (ctx->opts.use_huge_pages)
        INFO("tensor: adaptive huge-page granule engaged (2MB reads off carveouts, 4KB near them)");

    /* Locate the reserved-memory device-tree node. */
    const char *base = NULL;
    for (size_t i = 0; i < sizeof(DT_BASES) / sizeof(DT_BASES[0]); i++) {
        if (access(DT_BASES[i], F_OK) == 0) { base = DT_BASES[i]; break; }
    }
    if (!base) {
        ERR("tensor: reserved-memory device-tree node not found; cannot build avoid-list");
        return ENOENT;
    }

    DIR *d = opendir(base);
    if (!d) RETURN_ERRNO("tensor: opendir reserved-memory");

    struct dirent *de;
    int n_nomap = 0;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') continue;

        char node_dir[1024];
        snprintf(node_dir, sizeof(node_dir), "%s/%s", base, de->d_name);

        /*
         * Conservative policy: every ENABLED no-map carveout is treated as potentially
         * S2MPU-fatal. no-map regions that proved benign (log/ramoops buffers) only
         * cost us a small read margin; the fatal firmware regions are all no-map.
         *
         * BUT a status="disabled" node was never claimed by the kernel (absent from
         * reserved_mem[]) so its range is ordinary readable RAM, not a carveout. Avoiding it
         * pattern-fills live data — e.g. s2m@c0000000 (disabled, 74MB) holds the KV dma_buf
         * descriptors and reads fine. Skip disabled nodes; keep every enabled no-map node.
         */
        if (!node_has(node_dir, "no-map")) continue;
        if (node_disabled(node_dir)) {
            DBG("tensor: skip disabled reserved-memory node %s (not carved out, readable RAM)", de->d_name);
            continue;
        }
        n_nomap++;
        parse_node_reg(ctx, node_dir, de->d_name);
    }
    closedir(d);

    /* Whitelist CMA (readable movable RAM that /proc/iomem labels "reserved") BEFORE the iomem scan,
     * so add_iomem_reserved() can skip it. CMA placement is dynamic - read live from the kernel. */
    tensor_init_cma(ctx);

    /* Second source: reserved ranges nested inside System RAM (not in the device tree). */
    add_iomem_reserved(ctx);

    /* Force-read override: kernel page tables are MMU-readable and must never be pattern-filled,
     * even when they land inside a carveout margin. Built live from swapper_pg_dir (EL1, no EL2). */
    tensor_init_pt_allowset(ctx);

    INFO("tensor: %d no-map carveouts + iomem reserved -> %d avoid-ranges (fw-margin 0x%llx, resv-margin 0x%llx, kimage forced-read)",
         n_nomap, g_nranges, (unsigned long long)TENSOR_FW_MARGIN, (unsigned long long)TENSOR_RESV_MARGIN);

    if (g_nranges == 0)
        WARN("tensor: no carveout ranges parsed; dump may hit S2MPU and reboot");

    return 1;
}

/*
 * tensor_is_protected_page() - True if a physical page must NOT be read.
 * @page_start: Physical base address of the dump chunk.
 *
 * Mirrors qualcomm_is_secure_page(): dump.c calls this and pattern-fills instead of
 * reading when it returns true, keeping the LiME physical layout intact.
 */
/*
 * Kernel-image physical window [g_kimage_lo, g_kimage_hi). The kernel executes from its own image,
 * so those pages can NEVER be S2MPU-protected -> always safe to read, even when they fall inside an
 * avoid-range or its margin. This is what recovers init_task, swapper_pg_dir and the kernel .data
 * that the old blanket 2 MB margin pattern-filled. _text_PA = _text - kimage_voffset.
 */
static uint64_t g_kimage_lo = 0, g_kimage_hi = 0;
static int g_kimage_ready = 0;

static void tensor_init_kimage(void) {
    g_kimage_ready = 1;   /* try once; on failure the override is simply inactive */
    uintptr_t text = 0, end = 0, vo_addr = 0;
    unsigned long addr;
    char line[512], name[128];
    FILE *fp = fopen("/proc/kallsyms", "r");
    if (!fp) return;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "%lx %*c %127s", &addr, name) != 2) continue;
        if      (!strcmp(name, "_text"))          text = (uintptr_t)addr;
        else if (!strcmp(name, "_end"))           end = (uintptr_t)addr;
        else if (!strcmp(name, "kimage_voffset")) vo_addr = (uintptr_t)addr;
    }
    fclose(fp);
    if (!text || !end || !vo_addr || end <= text) return;
    const unsigned char *d = NULL;
    if (read_kernel_memory(vo_addr, 8, &d) < 0 || !d) return;
    uint64_t voff = *(const uint64_t *)d;
    g_kimage_lo = (uint64_t)text - voff;
    g_kimage_hi = (uint64_t)end  - voff;
    INFO("tensor: kernel-image PA [0x%llx, 0x%llx) forced readable (never S2MPU-protected)",
         (unsigned long long)g_kimage_lo, (unsigned long long)g_kimage_hi);
}

bool tensor_is_protected_page(uintptr_t page_start) {
    if (!g_kimage_ready) tensor_init_kimage();
    /* kimage override: kernel image is CPU-executed, never S2MPU-protected -> always read it. */
    if (g_kimage_lo && (uint64_t)page_start >= g_kimage_lo && (uint64_t)page_start < g_kimage_hi)
        return false;
    /* page-table override: kernel PT pages are MMU-walked, so always readable -> force-read even if
     * they fall inside a carveout margin (fixes vmemmap/swapper over-fill; keeps address translation
     * intact in the dump). Built in check_init_tensor from swapper_pg_dir. */
    if (g_pt_ready && tensor_pt_is_pagetable((uint64_t)page_start))
        return false;
    /* Per-page CMA decision via the allocation bitmap: a cma_alloc'd (device) page is S2MPU-protected
     * -> fill; a free/movable page (where swapped anon/zsmalloc data lives) is CPU-readable -> read. */
    int cs = cma_status((uint64_t)page_start);
    if (cs == CMA_DEVICE)   return true;
    if (cs == CMA_READABLE) return false;
    for (int i = 0; i < g_nranges; i++)
        if ((uint64_t)page_start >= g_ranges[i].start &&
            (uint64_t)page_start <= g_ranges[i].end)
            return true;
    return false;
}

/*
 * tensor_range_overlaps() - True if inclusive [start, end] intersects any avoid-range.
 *
 * Used by dump.c's adaptive-granule path: a huge (2 MB) chunk that touches a carveout
 * must be reprocessed at PAGE_SIZE rather than read wholesale (an S2MPU read = reboot).
 * Note start<=end always holds (dump.c clamps chunk_end >= chunk_start).
 */
bool tensor_range_overlaps(uintptr_t start, uintptr_t end) {
    /* A huge (2 MB) chunk that spans a CMA device (protected) page must be reprocessed at PAGE_SIZE,
     * so the per-page bitmap in tensor_is_protected_page() fills only the protected pages and reads
     * the movable/free ones. All-readable CMA chunks fall through and can be read wholesale. */
    if (cma_range_has_device((uint64_t)start, (uint64_t)end)) return true;
    for (int i = 0; i < g_nranges; i++)
        if ((uint64_t)start <= g_ranges[i].end &&
            (uint64_t)end   >= g_ranges[i].start)
            return true;
    return false;
}
