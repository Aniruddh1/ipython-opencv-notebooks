function [vol_pL phi_urad]  = dispenser_model_4(waveform,
				    smpl_clk1_ns,
				    vi1_v)
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

  x0 = smpl_clk1_ns - 500; 
  a0 = 0.00;
  a00 = 0.0003;

  x1 =25 -  (vi1_v + 5);
  a1 = 0.253;
  a11 = 0.0;

  a01 = 0.00000;

  vol_pL  = 0.0;
  vol_pL += a0*x0  + a00*x0^2;
  vol_pL += a1*x1 + a11*x1^2;  
  vol_pL += a01*x0*x1;

  sig =  1.0 / (1.0 + exp(-(vol_pL - 2.0)/0.3)*15.10);

  if (sig > 0.95) || (vi1_v > 20)
    vol_pL = 10;
  else
    if (vol_pL < 0.6)
      vol_pL = 0.6;
    endif

    %% add some IID noise
    vol_pL += noise_pL*randn(1);
  endif

  theta = pi/4;
  rot = [cos(theta) -sin(theta);
	 sin(theta) cos(theta)];

  xr = rot*[x0/20 x1]';

  b00 = 0.0002;
  b11 = 0.00;
  phi_urad = b00 * xr(1)^2 + b11*xr(2)^2;

  phi_urad += 0.005*randn(1,1);

endfunction
