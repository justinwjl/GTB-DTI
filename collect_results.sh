#!/usr/bin/env bash
# Collect every finished run into one long CSV.
#
#   ./collect_results.sh [output.csv]
#
set -u

OUT=${1:-results_summary.csv}
echo "dataset,model,run,metric,mean,std,n_folds" > "$OUT"

for f in experiment/*/*/results.csv; do
    [ -e "$f" ] || continue
    run=$(basename "$(dirname "$f")")                    # ModelClass-YYYYmmdd-HHMMSS
    dataset=$(basename "$(dirname "$(dirname "$f")")")
    model=${run%-*-*}
    awk -F, -v d="$dataset" -v m="$model" -v r="$run" '
        NR > 1 && NF > 3 {
            n = 0
            for (i = 2; i <= NF - 2; i++) if ($i != "" && $i != "nan") n++
            print d "," m "," r "," $1 "," $(NF - 1) "," $NF "," n
        }' "$f" >> "$OUT"
done

echo "$(($(wc -l < "$OUT") - 1)) rows -> $OUT"
awk -F, 'NR > 1 && $7 != "" && $7 < 5 {print "  incomplete: " $1 " " $2 " " $4 " over " $7 " folds"}' "$OUT"
