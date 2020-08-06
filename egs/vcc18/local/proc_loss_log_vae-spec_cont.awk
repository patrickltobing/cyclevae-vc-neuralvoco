# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

#32 5.212066
#33 (+-
#34 0.443200)
#35 dB
#36 4.984742
#37 (+-
#38 0.405636)
#39 dB
#40 ;
#41 1.937963
#42 1.936878
#43 6.783503
#44 (+-
#45 0.621711)
#46 dB
#47 6.048465
#48 (+-
#49 0.497230)
#50 dB
#51 0.062046
#52 (+-
#53 0.007049)
#54 0.847297
#55 (+-
#56 0.034724)
#57 ;;
#
#31 5.413379
#32 (+-
#33 0.383929)
#34 dB
#35 5.090335
#36 (+-
#37 0.361550)
#38 dB
#
#63 5.791484
#64 (+-
#65 0.465184)
#66 dB
#67 5.364550
#68 (+-
#69 0.430153)
#70 dB

BEGIN {
    min_idx=0;
    min_val7=9999.9;
    min_val9=9999.9;
    min2_val7=9999.9;
    min2_db_pow=9999.9;
    min2_val9=9999.9;
    min2_db=9999.9;
} {
    if ($6=="average" && $7=="evaluation") {
        split($5,str1,")");
        split(str1[1],str2,":");
        idx=str2[2];
        printf "%d: %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s dB %.3f (+- %s dB ;; "\
                  "%.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f %.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f ;;\n", \
                      idx, tmp_trn1, tmp_trn2, tmp_trn3, tmp_trn4, tmp_trn5, tmp_trn6, tmp_trn7, tmp_trn8, \
                        $36, $38, $40, $42, $45, $46, $47, $49, $51, $53, $55, $58;
        split($49,a,")")
        std_db_pow=a[1]
        split($53,a,")")
        std_db=a[1]
        if ($47+$51<=min_val7+min_val9) {
            min_idx=idx;
            min_val1=$36;
            min_val2=$38;
            min_val3=$40;
            min_val4=$42;
            min_val5=$45;
            min_val6=$46;
            min_val7=$47;
            min_val8=$49;
            min_val9=$51;
            min_val10=$53;
            min_val11=$55;
            min_val12=$58;
            min_trn1=tmp_trn1
            min_trn2=tmp_trn2
            min_trn3=tmp_trn3
            min_trn4=tmp_trn4
            min_trn5=tmp_trn5
            min_trn6=tmp_trn6
            min_trn7=tmp_trn7
            min_trn8=tmp_trn8
        }
        if ($47+std_db_pow+$51+std_db<=min2_val7+min2_db_pow+min2_val9+min2_db) {
            min2_idx=idx;
            min2_val1=$36;
            min2_val2=$38;
            min2_val3=$40;
            min2_val4=$42;
            min2_val5=$45;
            min2_val6=$46;
            min2_val7=$47;
            min2_val8=$49;
            min2_db_pow=std_db_pow;
            min2_val9=$51;
            min2_val10=$53;
            min2_db=std_db;
            min2_val11=$55;
            min2_val12=$58;
            min2_trn1=tmp_trn1
            min2_trn2=tmp_trn2
            min2_trn3=tmp_trn3
            min2_trn4=tmp_trn4
            min2_trn5=tmp_trn5
            min2_trn6=tmp_trn6
            min2_trn7=tmp_trn7
            min2_trn8=tmp_trn8
        }
    } else if ($6=="average" && $7=="optimization") {
        tmp_trn1=$35;
        tmp_trn2=$37;
        tmp_trn3=$39;
        tmp_trn4=$41;
        tmp_trn5=$67;
        tmp_trn6=$69;
        tmp_trn7=$71;
        tmp_trn8=$73;
    }
} END {
    printf "#min=%d: %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s dB %.3f (+- %s dB ;; "\
              "%.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f %.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f ;;\n", \
                   min_idx, min_trn1, min_trn2, min_trn3, min_trn4, min_trn5, min_trn6, min_trn7, min_trn8, \
                       min_val1, min_val2, min_val3, min_val4, min_val5, min_val6, min_val7, min_val8, min_val9, min_val10, min_val11, min_val12;
    printf "#min=%d: %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s dB %.3f (+- %s dB ;; "\
              "%.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f %.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f ;;\n", \
                   min2_idx, min2_trn1, min2_trn2, min2_trn3, min2_trn4, min2_trn5, min2_trn6, min2_trn7, min2_trn8, \
                       min2_val1, min2_val2, min2_val3, min2_val4, min2_val5, min2_val6, min2_val7, min2_val8, min2_val9, min2_val10, min2_val11, min2_val12;
}

