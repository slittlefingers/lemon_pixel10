#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <unistd.h>

#include "../lemon.h"

/* LEMON's kernel-memory read oracle (defined in mem.c); used to read kimage_voffset's value. */
extern int read_kernel_memory(const uintptr_t addr, const size_t size, const unsigned char **restrict data);

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
 * CMA whitelist - the fix for "reserved but actually readable" over-avoidance.
 *
 * CMA areas (Contiguous Memory Allocator: video vframe/vstream pools, dma pools, pkvm_iommu_cma) are
 * DYNAMICALLY-placed movable RAM. They appear as "reserved" in /proc/iomem, so add_iomem_reserved()
 * used to pattern-fill them - but they are ordinary CPU-readable RAM that the page allocator hands to
 * movable/zsmalloc allocations (swap compression). Filling them destroyed readable kernel/user data
 * (e.g. the swapped scudo heap that holds the runtime name->buffer map). We read the kernel's
 * cma_areas[] at init and whitelist their runtime ranges. CMA placement is per-boot dynamic, so we
 * MUST read it live each capture - never hardcode.
 *
 * struct cma: base_pfn@0, count@8 are the stable first two fields (units: pages). The array stride =
 * sizeof(struct cma) is config-dependent (spinlock_t grows under lockdep). Verify for your kernel:
 *     pahole -C cma vmlinux        (or from the vmlinux BTF)
 * A wrong stride is caught by the per-entry sanity check below - bad entries are skipped, so at worst
 * we fall back to the old conservative behaviour (never a fatal read).
 */
#ifndef TENSOR_CMA_STRIDE
#define TENSOR_CMA_STRIDE 136   /* sizeof(struct cma) on 6.6 android15-4k; set from BTF/pahole per build */
#endif
#define TENSOR_MAX_CMA 32

struct tensor_cma { uint64_t start; uint64_t end; };   /* inclusive phys range, CPU-readable */
static struct tensor_cma g_cma[TENSOR_MAX_CMA];
static int g_ncma = 0;

/* True if a physical address lies in a whitelisted CMA (readable) region. */
static bool in_cma(uint64_t pa) {
    for (int i = 0; i < g_ncma; i++)
        if (pa >= g_cma[i].start && pa <= g_cma[i].end) return true;
    return false;
}

/*
 * tensor_init_cma() - read the kernel's cma_areas[] and record each CMA region as readable.
 * Uses the same /proc/kallsyms + read_kernel_memory oracle as tensor_init_kimage(). Dynamic per boot.
 */
static void tensor_init_cma(void) {
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
    if (!areas || !count_addr) { DBG("tensor: cma_areas/cma_area_count not in kallsyms; CMA not whitelisted"); return; }

    const unsigned char *d = NULL;
    if (read_kernel_memory(count_addr, 4, &d) < 0 || !d) return;
    unsigned int n = *(const unsigned int *)d;
    if (n > TENSOR_MAX_CMA) n = TENSOR_MAX_CMA;

    for (unsigned int i = 0; i < n; i++) {
        const unsigned char *e = NULL;
        if (read_kernel_memory(areas + (uintptr_t)i * TENSOR_CMA_STRIDE, 16, &e) < 0 || !e) continue;
        uint64_t base_pfn = *(const uint64_t *)e;
        uint64_t cnt      = *(const uint64_t *)(e + 8);
        /* Sanity-guard a wrong stride / empty slot: plausible pfn and page count only. */
        if (base_pfn == 0 || cnt == 0 || base_pfn > 0x2000000ULL || cnt > 0x1000000ULL) continue;
        if (g_ncma >= TENSOR_MAX_CMA) break;
        g_cma[g_ncma].start = base_pfn << 12;
        g_cma[g_ncma].end   = ((base_pfn + cnt) << 12) - 1;
        INFO("tensor: CMA readable phys 0x%llx-0x%llx (%llu MB) whitelisted",
             (unsigned long long)g_cma[g_ncma].start, (unsigned long long)g_cma[g_ncma].end,
             (unsigned long long)(cnt >> 8));
        g_ncma++;
    }
    INFO("tensor: %d CMA region(s) whitelisted from kernel cma_areas[] (dynamic, read per capture)", g_ncma);
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
            /* CMA pools show as "reserved" but are readable movable RAM -> do not avoid them. */
            if (in_cma(s) && in_cma(e)) { skipped_cma++; continue; }
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
         * Conservative policy: every no-map carveout is treated as potentially
         * S2MPU-fatal. no-map regions that proved benign (log/ramoops buffers) only
         * cost us a small read margin; the fatal firmware regions are all no-map.
         * Refine to a firmware-only deny set later if completeness matters.
         */
        if (!node_has(node_dir, "no-map")) continue;
        n_nomap++;
        parse_node_reg(ctx, node_dir, de->d_name);
    }
    closedir(d);

    /* Whitelist CMA (readable movable RAM that /proc/iomem labels "reserved") BEFORE the iomem scan,
     * so add_iomem_reserved() can skip it. CMA placement is dynamic - read live from the kernel. */
    tensor_init_cma();

    /* Second source: reserved ranges nested inside System RAM (not in the device tree). */
    add_iomem_reserved(ctx);

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
    /* CMA override: movable RAM (video/dma pools) the allocator uses is CPU-readable -> never fill it,
     * even if a firmware carveout's margin spilled onto it. (g_ncma from tensor_init_cma().) */
    if (in_cma((uint64_t)page_start))
        return false;
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
    for (int i = 0; i < g_nranges; i++)
        if ((uint64_t)start <= g_ranges[i].end &&
            (uint64_t)end   >= g_ranges[i].start)
            return true;
    return false;
}
