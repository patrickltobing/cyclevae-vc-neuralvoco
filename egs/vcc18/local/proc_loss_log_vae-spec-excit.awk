# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

BEGIN {
    min_idx=0;
    min_val7=9999.9;
    min_val9=9999.9;
    min_val13=9999.9;
    min_val15=9999.9;
    min2_val7=9999.9;
    min2_hz_rec=9999.9;
    min2_val9=9999.9;
    min2_hz_cv=9999.9;
    min2_val13=9999.9;
    min2_db_pow=9999.9;
    min2_val15=9999.9;
    min2_db=9999.9;
    min3_val13=9999.9;
    min3_db_pow=9999.9;
    min3_val15=9999.9;
    min3_db=9999.9;
    min4_val13=9999.9;
    min4_val15=9999.9;
} {
    if ($2=="average" && $3=="evaluation") {
        split($1,str1,")");
        split(str1[1],str2,":");
        idx=str2[2];
        printf "%d: %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz ;; "\
                  "%.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f %.3f %.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f ;;\n", \
                      idx, tmp_trn1, tmp_trn2, tmp_trn3, tmp_trn4, tmp_trn5, tmp_trn6, tmp_trn7, tmp_trn8, tmp_trn9, tmp_trn10, tmp_trn11, tmp_trn12, tmp_trn13, tmp_trn14, tmp_trn15, tmp_trn16, tmp_trn17, tmp_trn18, \
                        $43, $45, $47, $49, $52, $54, $61, $63, $65, $67, $88, $89, $90, $92, $94, $96, $114, $117;
        split($63,a,")")
        std_hz_rec=a[1]
        split($67,a,")")
        std_hz_cv=a[1]
        split($92,a,")")
        std_db_pow=a[1]
        split($96,a,")")
        std_db=a[1]
        if ($61+$65+$90+$94<=min_val7+min_val9+min_val13+min_val15) {
            min_idx=idx;
            min_val1=$43;
            min_val2=$45;
            min_val3=$47;
            min_val4=$49;
            min_val5=$52;
            min_val6=$54;
            min_val7=$61;
            min_val8=$63;
            min_val9=$65;
            min_val10=$67;
            min_val11=$88;
            min_val12=$89;
            min_val13=$90;
            min_val14=$92;
            min_val15=$94;
            min_val16=$96;
            min_val17=$114;
            min_val18=$117;
            min_trn1=tmp_trn1
            min_trn2=tmp_trn2
            min_trn3=tmp_trn3
            min_trn4=tmp_trn4
            min_trn5=tmp_trn5
            min_trn6=tmp_trn6
            min_trn7=tmp_trn7
            min_trn8=tmp_trn8
            min_trn9=tmp_trn9
            min_trn10=tmp_trn10
            min_trn11=tmp_trn11
            min_trn12=tmp_trn12
            min_trn13=tmp_trn13
            min_trn14=tmp_trn14
            min_trn15=tmp_trn15
            min_trn16=tmp_trn16
            min_trn17=tmp_trn17
            min_trn18=tmp_trn18
        }
        if ($61+std_hz_rec+$65+std_hz_cv+$90+std_db_pow+$94+std_db<=min2_val7+min2_hz_rec+min2_val9+min2_hz_cv+min2_val13+min2_db_pow+min2_val15+min2_db) {
            min2_idx=idx;
            min2_val1=$43;
            min2_val2=$45;
            min2_val3=$47;
            min2_val4=$49;
            min2_val5=$52;
            min2_val6=$54;
            min2_val7=$61;
            min2_val8=$63;
            min2_hz_rec=std_hz_rec;
            min2_val9=$65;
            min2_val10=$67;
            min2_hz_cv=std_hz_cv;
            min2_val11=$88;
            min2_val12=$89;
            min2_val13=$90;
            min2_val14=$92;
            min2_db_pow=std_db_pow;
            min2_val15=$94;
            min2_val16=$96;
            min2_db=std_db;
            min2_val17=$114;
            min2_val18=$117;
            min2_trn1=tmp_trn1
            min2_trn2=tmp_trn2
            min2_trn3=tmp_trn3
            min2_trn4=tmp_trn4
            min2_trn5=tmp_trn5
            min2_trn6=tmp_trn6
            min2_trn7=tmp_trn7
            min2_trn8=tmp_trn8
            min2_trn9=tmp_trn9
            min2_trn10=tmp_trn10
            min2_trn11=tmp_trn11
            min2_trn12=tmp_trn12
            min2_trn13=tmp_trn13
            min2_trn14=tmp_trn14
            min2_trn15=tmp_trn15
            min2_trn16=tmp_trn16
            min2_trn17=tmp_trn17
            min2_trn18=tmp_trn18
        }
        if ($90+std_db_pow+$94+std_db<=min3_val13+min3_db_pow+min3_val15+min3_db) {
            min3_idx=idx;
            min3_val1=$43;
            min3_val2=$45;
            min3_val3=$47;
            min3_val4=$49;
            min3_val5=$52;
            min3_val6=$54;
            min3_val7=$61;
            min3_val8=$63;
            min3_val9=$65;
            min3_val10=$67;
            min3_val11=$88;
            min3_val12=$89;
            min3_val13=$90;
            min3_val14=$92;
            min3_db_pow=std_db_pow;
            min3_val15=$94;
            min3_val16=$96;
            min3_db=std_db;
            min3_val17=$114;
            min3_val18=$117;
            min3_trn1=tmp_trn1
            min3_trn2=tmp_trn2
            min3_trn3=tmp_trn3
            min3_trn4=tmp_trn4
            min3_trn5=tmp_trn5
            min3_trn6=tmp_trn6
            min3_trn7=tmp_trn7
            min3_trn8=tmp_trn8
            min3_trn9=tmp_trn9
            min3_trn10=tmp_trn10
            min3_trn11=tmp_trn11
            min3_trn12=tmp_trn12
            min3_trn13=tmp_trn13
            min3_trn14=tmp_trn14
            min3_trn15=tmp_trn15
            min3_trn16=tmp_trn16
            min3_trn17=tmp_trn17
            min3_trn18=tmp_trn18
        }
        if ($90+$94<=min4_val13+min4_val15) {
            min4_idx=idx;
            min4_val1=$43;
            min4_val2=$45;
            min4_val3=$47;
            min4_val4=$49;
            min4_val5=$52;
            min4_val6=$54;
            min4_val7=$61;
            min4_val8=$63;
            min4_val9=$65;
            min4_val10=$67;
            min4_val11=$88;
            min4_val12=$89;
            min4_val13=$90;
            min4_val14=$92;
            min4_val15=$94;
            min4_val16=$96;
            min4_val17=$114;
            min4_val18=$117;
            min4_trn1=tmp_trn1
            min4_trn2=tmp_trn2
            min4_trn3=tmp_trn3
            min4_trn4=tmp_trn4
            min4_trn5=tmp_trn5
            min4_trn6=tmp_trn6
            min4_trn7=tmp_trn7
            min4_trn8=tmp_trn8
            min4_trn9=tmp_trn9
            min4_trn10=tmp_trn10
            min4_trn11=tmp_trn11
            min4_trn12=tmp_trn12
            min4_trn13=tmp_trn13
            min4_trn14=tmp_trn14
            min4_trn15=tmp_trn15
            min4_trn16=tmp_trn16
            min4_trn17=tmp_trn17
            min4_trn18=tmp_trn18
        }
    } else if ($2=="average" && $3=="optimization") {
        tmp_trn1=$42;
        tmp_trn2=$44;
        tmp_trn3=$46;
        tmp_trn4=$48;
        tmp_trn5=$51;
        tmp_trn6=$53;
        tmp_trn7=$60;
        tmp_trn8=$62;
        tmp_trn9=$64;
        tmp_trn10=$66;
        tmp_trn11=$121;
        tmp_trn12=$123;
        tmp_trn13=$125;
        tmp_trn14=$127;
        tmp_trn15=$130;
        tmp_trn16=$132;
        tmp_trn17=$134;
        tmp_trn18=$136;
    }
} END {
    printf "#min=%d: %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz ;; "\
              "%.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f %.3f %.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f ;;\n", \
                   min_idx, min_trn1, min_trn2, min_trn3, min_trn4, min_trn5, min_trn6, min_trn7, min_trn8, min_trn9, min_trn10, min_trn11, min_trn12, min_trn13, min_trn14, min_trn15, min_trn16, min_trn17, min_trn18, \
                       min_val1, min_val2, min_val3, min_val4, min_val5, min_val6, min_val7, min_val8, min_val9, min_val10, min_val11, min_val12, min_val13, min_val14, min_val15, min_val16, min_val17, min_val18;
    printf "#min=%d: %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz ;; "\
              "%.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f %.3f %.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f ;;\n", \
                   min2_idx, min2_trn1, min2_trn2, min2_trn3, min2_trn4, min2_trn5, min2_trn6, min2_trn7, min2_trn8, min2_trn9, min2_trn10, min2_trn11, min2_trn12, min2_trn13, min2_trn14, min2_trn15, min2_trn16, min2_trn17, min2_trn18,\
                       min2_val1, min2_val2, min2_val3, min2_val4, min2_val5, min2_val6, min2_val7, min2_val8, min2_val9, min2_val10, min2_val11, min2_val12, min2_val13, min2_val14, min2_val15, min2_val16, min2_val17, min2_val18;
    printf "#min=%d: %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz ;; "\
              "%.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f %.3f %.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f ;;\n", \
                   min3_idx, min3_trn1, min3_trn2, min3_trn3, min3_trn4, min3_trn5, min3_trn6, min3_trn7, min3_trn8, min3_trn9, min3_trn10, min3_trn11, min3_trn12, min3_trn13, min3_trn14, min3_trn15, min3_trn16, min3_trn17, min3_trn18,\
                       min3_val1, min3_val2, min3_val3, min3_val4, min3_val5, min3_val6, min3_val7, min3_val8, min3_val9, min3_val10, min3_val11, min3_val12, min3_val13, min3_val14, min3_val15, min3_val16, min3_val17, min3_val18;
    printf "#min=%d: %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz ;; "\
              "%.3f (+- %s dB %.3f (+- %s dB , %.3f (+- %s %% %.3f (+- %s Hz %.3f (+- %s Hz , %.3f %.3f %.3f (+- %s dB %.3f (+- %s dB , %.3f %.3f ;;\n", \
                   min4_idx, min4_trn1, min4_trn2, min4_trn3, min4_trn4, min4_trn5, min4_trn6, min4_trn7, min4_trn8, min4_trn9, min4_trn10, min4_trn11, min4_trn12, min4_trn13, min4_trn14, min4_trn15, min4_trn16, min4_trn17, min4_trn18,\
                       min4_val1, min4_val2, min4_val3, min4_val4, min4_val5, min4_val6, min4_val7, min4_val8, min4_val9, min4_val10, min4_val11, min4_val12, min4_val13, min4_val14, min4_val15, min4_val16, min4_val17, min4_val18;
}

