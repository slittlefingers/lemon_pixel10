#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <unistd.h>

#include "../lemon.h"

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
 * Safety margin applied on both sides of every carveout.
 *
 * The fatal read observed on Pixel 10 (Tensor G5) was the System-RAM page right before
 * tpu_fw@9fc00000, i.e. S2MPU protects slightly beyond the device-tree boundary (or a
 * prefetch spills in). 2 MB is a conservative cover for the S2MPU granularity.
 * TUNE THIS against the device via the bisection step in docs/test-runbook.md.
 */
#define TENSOR_SAFETY_MARGIN 0x200000ULL   /* 2 MB */

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
 * add_range() - record one [base, base+size) carveout, expanded by TENSOR_SAFETY_MARGIN.
 * Stored inclusive [start, end]. Margin is clamped at 0 on the low side.
 */
static void add_range(const struct lemon_ctx *restrict ctx, const char *name, uint64_t base, uint64_t size) {
    if (size == 0 || g_nranges >= TENSOR_MAX_CARVEOUTS) return;

    uint64_t lo = (base > TENSOR_SAFETY_MARGIN) ? base - TENSOR_SAFETY_MARGIN : 0;
    uint64_t hi = base + size - 1 + TENSOR_SAFETY_MARGIN;

    g_ranges[g_nranges].start = lo;
    g_ranges[g_nranges].end   = hi;
    snprintf(g_ranges[g_nranges].name, sizeof(g_ranges[g_nranges].name), "%s", name);
    g_nranges++;

    DBG("tensor: avoid %-28s phys 0x%llx-0x%llx (carveout 0x%llx+0x%llx +/-margin)",
        name, (unsigned long long)lo, (unsigned long long)hi,
        (unsigned long long)base, (unsigned long long)size);
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
        add_range(ctx, name, be64(buf + off), be64(buf + off + 8));
}

/*
 * add_iomem_reserved() - add "reserved" ranges nested inside System RAM (from /proc/iomem).
 *
 * Second danger source (see recon-findings.md section 10): LEMON dumps "System RAM"
 * ranges but does NOT subtract the reserved children nested within them (iomem.c
 * range_subtract() is unused). On Tensor those nested reserved regions - e.g.
 * 0xacc000000-0xaf83fffff inside the 0x880000000-0xaffffffff bank - are S2MPU-protected
 * and fatal to read, yet they are absent from the device-tree reserved-memory node.
 * Add every /proc/iomem "reserved" range to the avoid-list. Conservative: a few reserved
 * ranges read fine, but skipping them only costs a small margin.
 */
static void add_iomem_reserved(const struct lemon_ctx *restrict ctx) {
    FILE *fp = fopen("/proc/iomem", "r");
    if (!fp) { DBG("tensor: cannot open /proc/iomem for reserved scan"); return; }

    char line[512];
    int added = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long s, e;
        int pos = 0;
        /* "  <start>-<end> : <label>" ; %llx skips the indentation whitespace. */
        if (sscanf(line, "%llx-%llx : %n", &s, &e, &pos) < 2 || pos == 0)
            continue;
        if (strcasestr(line + pos, "reserved")) {
            add_range(ctx, "iomem-reserved", s, e - s + 1);
            added++;
        }
    }
    fclose(fp);
    INFO("tensor: +%d /proc/iomem reserved ranges added to avoid-list", added);
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

    /* Second source: reserved ranges nested inside System RAM (not in the device tree). */
    add_iomem_reserved(ctx);

    INFO("tensor: %d no-map carveouts + iomem reserved -> %d avoid-ranges (margin 0x%llx)",
         n_nomap, g_nranges, (unsigned long long)TENSOR_SAFETY_MARGIN);

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
bool tensor_is_protected_page(uintptr_t page_start) {
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
