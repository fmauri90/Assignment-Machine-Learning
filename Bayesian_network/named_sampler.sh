#! /bin/bash
cat heart.dat > file_originale.dat
#awk -f numerazione.awk file_originale.dat > numerati.dat
awk 'BEGIN {FS=","; OFS=","; i = 0} {print i, $0; i=i+1}' file_originale.dat > numerati.dat 
#awk -f 0.awk numerati.dat > 0.dat
#awk -f 1.awk numerati.dat > 1.dat
#awk -F',' '{if ($2==0 || $2=="y") print $0;}' numerati.dat > 0.dat
#awk -F',' '{if ($2==1 || $2=="y") print $0;}' numerati.dat > 1.dat
awk -F',' '{if ($2==0) print $0;}' numerati.dat > file0.dat
awk -F',' '{if ($2==1) print $0;}' numerati.dat > file1.dat
shuf file0.dat > mix0.dat
shuf file1.dat > mix1.dat
awk 'BEGIN {FS=","; OFS=","; i = 1} {if (i<=27) print $0; i=i+1}' mix0.dat > primo.dat
awk 'BEGIN {FS=","; OFS=","; i = 1} {if (i>27) print $0; i=i+1}' mix0.dat > secondo.dat
awk 'BEGIN {FS=","; OFS=","; i = 1} {if (i<=73) print $0; i=i+1}' mix1.dat >> primo.dat
awk 'BEGIN {FS=","; OFS=","; i = 1} {if (i>73 && i<=145) print $0; i=i+1}' mix1.dat >> secondo.dat
awk 'BEGIN {FS=","; OFS=","; i = 1} {if (i==1) print "num", $0; i=i+1}' file_originale.dat > train.dat
awk 'BEGIN {FS=","; OFS=","; i = 1} {if (i==1) print "num", $0; i=i+1}' file_originale.dat > test.dat
shuf primo.dat >> train.dat 
shuf secondo.dat >> test.dat
#cat mix0.dat > file_tot.dat
#cat mix1.dat >> file_tot.dat
