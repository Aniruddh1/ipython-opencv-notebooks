fname = "20161020-222714_BitMap_7.dat";
f = fopen(fname, "rb");

r = 772;
c = 128;

[val,count] = fread(f,[r,c],"uchar");
% [val,count] = fread(f,[r,c],"uchar");
fclose(f);

%val       128x772                   790528  double
whos
max(val)

dat = zeros(r, c*8);

for i=1:rows(val)
   for j=1:c
      if (val(i,j) > 0)
         printf("[%d, %d] = %d\n", i, j, val(i,j));
      endif
   endfor
endfor

