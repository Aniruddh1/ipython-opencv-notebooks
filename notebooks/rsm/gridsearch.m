

%%%
%%% First lets look at the system response with a grid search
%%%
vi_hist = [];
clk_hist = [];
vol_hist = [];

best_vol = 9999.9;

num_evals = 0;

r = 0;
for vi = -5:1:30

  r += 1;

  c = 0;
  for clk = 100:10:990
    c += 1;
    %vol = dispenser_model_1("donotcare", clk, clk, vi, vi);
    %vol = dispenser_model_2("donotcare", clk, vi);
    [vol phi] = dispenser_model_4("donotcare", clk, vi);
    num_evals++;

    if (vol < best_vol)
      best_phi = phi;
      best_vol = vol;
      best_vi = vi;
      best_clk = clk;
    endif

    vi_hist(r,c)  = vi;
    clk_hist(r,c) = clk;
    vol_hist(r,c) = vol;
    phi_hist(r,c) = phi;
  endfor

endfor


printf("Grid Search Best Solution: %f %f => %f pL %f urad after %d evals\n", 
       best_clk, best_vi, best_vol, best_phi, num_evals);

figure(1)
surf(vi_hist, clk_hist, vol_hist);
xlabel("Vi")
ylabel("Sample Clock")
zlabel("Volume (pL)")
view([60 70])
colorbar()


figure(2)
surf(vi_hist, clk_hist, phi_hist);
xlabel("Vi")
ylabel("Sample Clock")
zlabel("Phi 3s (urad)")
view([60 70])
colorbar()
