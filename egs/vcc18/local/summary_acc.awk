BEGIN {
    n_read = 0
    n_data = 0
    mcd_rec = 0
    mcd_cyc = 0
    mcd_cv = 0
    lgd = 0
    rmse = 0
    cossim = 0
} {
    if ($0 == "== summary rec. acc. ==") {
        n_read++
    } else if (n_read == 1) {
        n_read++
    } else if (n_read == 2) {
        mcd_rec += $2
        n_read++
    } else if (n_read == 3) {
        n_read++
    } else if (n_read == 4) {
        n_read++
    } else if (n_read == 5) {
        mcd_cyc += $2
        n_read++
    } else if (n_read == 6) {
        n_read++
    } else if (n_read == 7) {
        lgd += $1
        n_read++
    } else if (n_read == 8) {
        n_read++
    } else if (n_read == 9) {
        mcd_cv += $2
        n_read++
    } else if (n_read == 10) {
        rmse += $2
        n_read++
    } else if (n_read == 11) {
        cossim += $2
        n_read++
        n_data++
    }
} END {
    printf "mcd_rec = %lf\n", mcd_rec/n_data
    printf "mcd_cyc = %lf\n", mcd_cyc/n_data
    printf "mcd_cv = %lf\n", mcd_cv/n_data
    printf "lgd = %lf\n", lgd/n_data
    printf "rmse = %lf\n", rmse/n_data
    printf "cossim = %lf\n", cossim/n_data
}
