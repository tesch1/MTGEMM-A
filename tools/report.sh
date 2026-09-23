#!/bin/bash
# Regenerates the markdown tables in results/final/tables.md from the raw benchmark files.
cd "$(dirname "$0")/.."
R=results/final
{
  for o in row col; do
    echo "### fp32 single thread, $o-major"
    echo
    python3 tools/table.py $o --paper $( [ $o = col ] && echo --eigen ) "Accel=$R/accel_$o.txt" "paper design=$R/paper_$o.txt" "MTGEMM-A*=$R/tuned_$o.txt"
    echo
  done
  for o in row col; do
    echo "### fp32 two threads (one per P-cluster SME unit), $o-major"
    echo
    python3 tools/table.py $o --paper-multi "Accel (all threads)=$R/accelN_$o.txt" "paper design 2T=$R/paper2_$o.txt" "MTGEMM-A 2T*=$R/mt2_$o.txt"
    echo
  done
  echo "### fp64 row-major"
  echo
  python3 tools/table.py row --paper-fp64 --paper-fp64-par "Accel 1T=$R/f64_accel_row.txt" "paper design=$R/f64_paper_row.txt" "MTGEMM-A*=$R/f64_tuned_row.txt" "Accel (all)=$R/f64_accelN_row.txt" "MTGEMM-A 2T=$R/f64_tuned2_row.txt"
  echo
  echo "### Irregular shapes (row-major, K = 25600)"
  echo
  python3 tools/table.py row "Accel=$R/irr_accel_row.txt" "paper design=$R/irr_paper_row.txt" "MTGEMM-A*=$R/irr_tuned_row.txt"
  echo
  echo "### Ablation from MTGEMM-A (one element off at a time)"
  python3 tools/ablation.py $R tuned abl_no_online abl_no_x4 abl_stack abl_fixed_blk abl_no_blk abl_k32x32 abl_pack1row abl_c_paper abl_c_direct abl_no_pf abl_no_pfA abl_no_pfB abl_no_pfC
  echo
  echo "### Ablation from the paper design"
  python3 tools/ablation.py $R paper pabl_no_online pabl_no_x4 pabl_stack pabl_fixed_blk pabl_no_blk pabl_k32x32
  echo
  X=results/ext
  echo "### LIBXSMM (column-major) and KleidiAI (row-major), one session"
  echo
  python3 tools/table.py col --paper --paper-lib=LIBXSMM "LIBXSMM=$X/libxsmm_col.txt" "Accel=$X/accel_col.txt" "MTGEMM-A*=$X/mt_col.txt"
  echo
  python3 tools/table.py row --paper --paper-lib=KleidiAI --paper-lib=OpenBLAS "KleidiAI=$X/kleidiai_row.txt" "Accel=$X/accel_row.txt" "MTGEMM-A*=$X/mt_row.txt"
} > $R/tables.md
echo "wrote $R/tables.md"
