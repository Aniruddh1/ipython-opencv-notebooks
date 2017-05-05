function vol_pL = dispenser_model_1(waveform,
				    smpl_clk1_ns, smpl_clk2_ns,
				    vi1_v, vi2_v)
  %% Setup
  noise_pL = 0.100;


  %% model
  %% we will ignore the _2 parameters
  if (smpl_clk1_ns < 100)
    smpl_clk1_ns = 100;
  endif

  if (smpl_clk1_ns > 990)
    smpl_clk1_ns = 990;
  endif

  if (vi1_v < -5)
    vi1_v = -5;
  endif

  if (vi1_v > 30)
    vi1_v = 30;
  endif

  x0 = smpl_clk1_ns - 100; 
  a0 = 0.0;
  a00 = 0.00001;

  x1 = vi1_v + 5;
  a1 = -0.03;
  a11 = 0.01;

  a01 = 0.0001;

  vol_pL  = 0.0;
  vol_pL += a0*(x0 - 300) + a00*(x0 - 300)^2;
  vol_pL += a1*(x1-25) + a11*(x1-25)^2;  
  vol_pL += a01*x0*x1;

  %% add some IID noise
  vol_pL += noise_pL*randn(1);

endfunction
