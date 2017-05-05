
NumSplsPerSet = 50;


%%%
%%% First lets look at the system response with a grid search
%%%
vi_hist = [];
clk_hist = [];
vol_hist = [];

best_vol = 9999.9;
num_evals = 0;

r = 0;
for vi = -5:2:30

  r += 1;

  c = 0;
  for clk = 100:2:990
    c += 1;
    
    vol = 0.0;
    for i=1:NumSplsPerSet
      vol += dispenser_model_1("donotcare", clk, clk, vi, vi);
      num_evals++;
    endfor
    vol /= NumSplsPerSet;

    if (vol < best_vol)
      best_vol = vol;
      best_vi = vi;
      best_clk = clk;
    endif

    vi_hist(r,c)  = vi;
    clk_hist(r,c) = clk;
    vol_hist(r,c) = vol;
  endfor

endfor


printf("Grid Search w/ Averaging Best Solution: %f %f => %f pL after %d evals\n", 
       best_clk, best_vi, best_vol, num_evals);

figure(1)
surf(vi_hist, clk_hist, vol_hist);
xlabel("Vi")
ylabel("Sample Clock")
zlabel("Volume (pL)")
