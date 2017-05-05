

stencil = [
           -1 -1;
           -1  1;
            1 -1;
            1  1;
           ];

alpha = [ 0.2 1]';

NumSplsPerSet = 50;
MaxNumEvals = 50000;

vol_tgt = 0.6;

step_width = [1.0 10];


start = [16 500];

Max_Step = [1.0 10];
Min_Lim = [-5 100];
Max_Lim = [30 990];

Tgt_Vol = 0.6;

% LS fit setup
M = ones(rows(stencil), 3);
M (:,2:3) = stencil;




num_evals = 0;
fit = [0 0 0]';
opt = start';
step = [0 0]';
cur = start';

iter = 0;

eval_hist = [];
a_hist = [];
f_hist = [];
step_hist = [];
J_hist = [];

best_objFunc = 99999.999;
objFunc      = 99999.999;
done = 0;

while ((num_evals < MaxNumEvals) && (!done))

  cur += step;

  if (cur(1) > Max_Lim(1)-2)
    cur(1) = Max_Lim(1)-2;
  endif

  if (cur(2) > Max_Lim(2)-20)
    cur(2) = Max_Lim(2)-20;
  endif

  if (cur(1) < Min_Lim(1)+2)
    cur(1) = Min_Lim(1)+2;
  endif

  if (cur(2) < Min_Lim(2)+20)
    cur(2) = Min_Lim(2)+20;
  endif


  % execute the res5CCD around the current setting
  beta = 1;
  for i=1:rows(stencil)
    vi  = cur(1) + stencil(i,1)*step_width(1);
    clk = cur(2) + stencil(i,2)*step_width(2);

    if (vi > Max_Lim(1))
      sc = (Max_Lim(1) - cur(1)) / step_width(1);
      if (sc < beta)
        beta = sc;
      endif
    endif

    if (clk > Max_Lim(2))
      sc = (Max_Lim(2) - cur(2)) / step_width(2);
      if (sc < beta)
        beta = sc;
      endif
    endif

    if (vi < Min_Lim(1))
      sc = (cur(1) - Min_Lim(1)) / step_width(1);
      if (sc < beta)
        beta = sc;
      endif
    endif

    if (clk < Min_Lim(2))
      sc = (cur(2) - Min_Lim(2)) / step_width(2);
      if (sc < beta)
        beta = sc;
      endif
    endif

  endfor

  for i=1:rows(stencil)
    vi  = cur(1) + stencil(i,1)*step_width(1)*beta;
    clk = cur(2) + stencil(i,2)*step_width(2)*beta;

    Mact (i,2:3) = [vi clk];

    vol(i,1) = 0.0;
    phi(i,1) = 0.0;

    for j=1:NumSplsPerSet
      [tmp1 tmp2] = dispenser_model_4("donotcare", clk, vi);
      vol(i,1) += tmp1;
      phi(i,1) += tmp2;
      num_evals++;
    endfor

    vol(i,1) /= NumSplsPerSet;
    phi(i,1) /= NumSplsPerSet;

  endfor

  iter++;
  eval_hist(iter,:) = cur;

  % fit the model
  a = M \ vol;
  a_hist(iter,:) = a;

  f = M \ phi;
  f_hist(iter,:) = f;

  dc = mean(Mact(:,2:3));
  sc = 1./max(Mact(:,2:3) - repmat(dc,rows(stencil),1));

  H  = [a(2)*a(2)   a(2)*a(3);
        a(2)*a(3)   a(3)*a(3)];

  H += [f(2)*f(2)   f(2)*f(3);
        f(2)*f(3)   f(3)*f(3)];

  J  = [a(2) a(3)]'*(a(1) - vol_tgt);
  J += [f(2) f(3)]'*(f(1));
  cur_sc= (cur - dc').*sc';
  J += H*cur_sc;

  J_hist(iter,:) = J;
  step = -1 * alpha .* (H \ J) ;

  step_hist(iter,:) = step;


  if (abs(step(1)) > Max_Step(1))
    step(1) = sign(step(1))*Max_Step(1);
  endif
  if (abs(step(2)) > Max_Step(2))
    step(2) = sign(step(2))*Max_Step(2);
  endif

  step_hist(iter,:) = step;

  prevObjFunc = objFunc;

  objFunc = (a(1) - vol_tgt)^2 + f(1)^2;
  if (objFunc < best_objFunc)
    best_objFunc = objFunc;
    best_vol = a(1);
    best_phi = f(1);
    best_clk = cur(2);
    best_vi  = cur(1);
  endif

  if (objFunc < 1e-5)
    done = 1;
  endif

endwhile


printf("RSM Search Best Solution: %f %f => %f pL after %d evals\n", 
       best_clk, best_vi, best_vol, num_evals);

figure(2)
plot(eval_hist(:,1), eval_hist(:,2), '*-')
xlabel("Vi")
ylabel("Sample Clock")

figure(3)
subplot(2,1,1)
plot(step_hist(:,1))
ylabel("dObj / dVi ")
subplot(2,1,2)
plot(step_hist(:,2))
ylabel("dObj / dClk ")

figure(4)
subplot(2,1,1)
plot(a_hist(:,1))
ylabel("Volume (pl)")
subplot(2,1,2)
plot(f_hist(:,1))
ylabel("Angle Std (rad)")
