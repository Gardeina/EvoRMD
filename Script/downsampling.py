import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def downsampling_from_raw(
    raw_csv,
    alpha=0.6,
    target_mods=("m6a", "m5c"),
    min_keep=1,
    seed=None,   # controls randomness of sampling
):
    """
    Downsampling logic consistent with the second script:
      - Directly compute summary statistics (sample counts per species × mod_type) from raw_csv
      - log_downsample_target compresses the scale of target modifications based on the average magnitude
        of "other modifications"
      - Allocation / compensation logic is fully aligned with your second script
      - Supports reproducible random sampling (controlled by `seed`)
    """
    # === 1. Read and normalize ===
    df = pd.read_csv(raw_csv)
    df["mod_type"] = df["mod_type"].astype(str).str.lower()
    df["species"] = df["species"].astype(str).str.strip().str.lower()

    # === 2. Automatically compute summary stats ===
    summary = (
        df.groupby(["species", "mod_type"], as_index=False)
          .size()
          .rename(columns={"size": "count"})
    )

    # === 3. Compute total counts and target downsampling size ===
    totals_by_mod = summary.groupby("mod_type", as_index=False)["count"].sum()
    totals_map = dict(zip(totals_by_mod["mod_type"], totals_by_mod["count"].astype(int)))
    # All non-target modification counts
    others_counts = [v for k, v in totals_map.items() if k not in target_mods]

    def log_downsample_target(n_total, others_counts, alpha):
        """Log-scale compression of target class size."""
        oc = np.asarray(others_counts, dtype=float)
        if oc.size == 0:
            # If there are no other modification types, keep all samples
            return int(n_total)
        # Geometric mean of "other modifications" (log-average in linear scale)
        L_mean = np.log10(np.exp(np.log(oc).mean()))
        # Interpolate between L_mean and log10(n_total) with factor alpha
        return int(10 ** (L_mean + alpha * (np.log10(n_total) - L_mean)))

    # Automatically compute target total size for each target modification
    target_total = {}
    for mod in target_mods:
        n_total = totals_map.get(mod, 0)
        if n_total > 0:
            target_total[mod] = log_downsample_target(n_total, others_counts, alpha)
        else:
            target_total[mod] = 0

    print(f"===> Automatically computed target totals (log-compressed): {target_total}")

    # === 4. Initialize RNG (for reproducible sampling) ===
    rng = np.random.default_rng(seed) if seed is not None else None

    # === 5. Sample per modification type ===
    keep_list = []
    for mod, mod_group in df.groupby("mod_type", sort=False):
        mod_l = mod.lower()

        # Non-target modifications: keep all samples
        if mod_l not in target_mods:
            keep_list.append(mod_group)
            continue

        target_total_mod = int(target_total[mod_l])
        if target_total_mod <= 0:
            # Target mod exists but requested size is 0 → skip
            continue

        # Available samples per species for this modification
        avail_stats = (
            mod_group.groupby("species", as_index=False)
                     .size()
                     .rename(columns={"size": "available"})
        )
        # Remove species with no samples (just in case)
        avail_stats = avail_stats[avail_stats["available"] > 0].copy()
        if avail_stats.empty:
            continue

        # --- Allocate target_total_mod across species (log-weighted by available count) ---
        tmp = avail_stats.copy()
        tmp["log_w"] = np.log(tmp["available"] + 1.0)
        tmp["prob"]  = tmp["log_w"] / tmp["log_w"].sum()
        tmp["raw_t"] = tmp["prob"] * target_total_mod

        # Floor allocation, then ensure at least min_keep if possible
        tmp["alloc"] = np.floor(tmp["raw_t"]).astype(int)
        tmp.loc[tmp["alloc"] < min_keep, "alloc"] = min_keep
        # Do not allocate more than available samples
        tmp["alloc"] = np.minimum(tmp["alloc"], tmp["available"])

        # Remainder after flooring and min_keep adjustment
        remainder = target_total_mod - int(tmp["alloc"].sum())
        if remainder > 0:
            # Species with remaining capacity
            tmp["room"] = tmp["available"] - tmp["alloc"]
            # Fractional part of raw target (used to prioritize species)
            tmp["frac"] = tmp["raw_t"] - np.floor(tmp["raw_t"])

            # Sort species by remaining capacity (room) and fractional part (frac)
            cand = tmp[tmp["room"] > 0].copy().sort_values(
                ["room", "frac"],
                ascending=[False, False]
            )

            # Distribute remainder greedily
            for idx in cand.index:
                if remainder <= 0:
                    break
                add = int(min(tmp.at[idx, "room"], remainder))
                tmp.at[idx, "alloc"] += add
                remainder -= add

        # --- Intra-species sampling ---
        sampled_idx = []
        for _, row in tmp.iterrows():
            sp, k = row["species"], int(row["alloc"])
            if k <= 0:
                continue
            sub_df = mod_group[mod_group["species"] == sp]

            if len(sub_df) <= k:
                # If samples are fewer than or equal to required, take all
                sampled_idx.extend(sub_df.index.tolist())
            else:
                # Draw a reproducible random_state if rng is provided
                rs = None if rng is None else int(rng.integers(0, 2**32 - 1))
                sampled_idx.extend(
                    sub_df.sample(n=k, random_state=rs).index.tolist()
                )

        # --- Global补充 sampling if total is still less than target_total_mod ---
        got = len(sampled_idx)
        if got < target_total_mod:
            need = target_total_mod - got
            remaining_pool = mod_group.drop(index=sampled_idx)
            if len(remaining_pool) > 0:
                rs_fill = None if rng is None else int(rng.integers(0, 2**32 - 1))
                add_n = min(need, len(remaining_pool))
                sampled_idx.extend(
                    remaining_pool.sample(n=add_n, random_state=rs_fill).index.tolist()
                )

        keep_list.append(df.loc[sampled_idx])

    # Concatenate all kept samples into the final balanced DataFrame
    balanced_df = pd.concat(keep_list, ignore_index=True)

    # === 6. Post-processing ===
    # Replace T with U in sequences, if the sequence column exists
    if "sequence" in balanced_df.columns:
        balanced_df["sequence"] = (
            balanced_df["sequence"]
            .astype(str)
            .str.replace("T", "U", regex=False)
        )

    # Encode mod_type labels as integers
    label_encoder = LabelEncoder()
    balanced_df["mod_type"] = label_encoder.fit_transform(
        balanced_df["mod_type"].astype(str)
    )

    print(f"✅ Downsampling completed (seed={seed}).")
    print("Label mapping:")
    for i, lab in enumerate(label_encoder.classes_):
        print(f"{i}: {lab}")
    print(balanced_df["mod_type"].value_counts())

    return balanced_df, label_encoder
