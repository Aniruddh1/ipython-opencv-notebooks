

% res3_stencil = [-1 -1;
%     -1  1;
%      1 -1;
%      1  1];
res3_stencil = [0 0;
    0  1;
     1 0];

%alpha = 30.5;
alpha = [5 100]'

NumSplsPerSet = 100;
MaxNumEvals = 100000;

step_width = [0.5 10];


start = [-4.5 980];

%Max_Step = [1.0 20];
max_step = [1.0 20];

Tgt_Vol = 0.6;

% LS fit setup
M = ones(rows(res3_stencil), columns(res3_stencil)+1);
M (:,2:end) = res3_stencil;



num_evals = 0;
best_vol = 9999.9;
fit = [99999.9 0 0]';
cur = start;
step = [0.0 0.0]';


iter = 0;
eval_hist = []
fit_hist = []

while ((num_evals < MaxNumEvals) && (best_vol > Tgt_Vol))

  cur += step';

  % execute the res3 around the current setting
  for i=1:rows(res3_stencil)
    vi  = cur(1) + res3_stencil(i,1)*step_width(1);
    clk = cur(2) + res3_stencil(i,2)*step_width(2);

    vol(i,1) = 0.0;

    for j=1:NumSplsPerSet
      vol(i,1) += dispenser_model_1("donotcare", clk, clk, vi, vi);
    num_evals++;
    endfor
    vol(i,1) /= NumSplsPerSet;

    if (vol(i,1) < best_vol)
      best_vol = vol(i,1);
      best_clk = clk;
      best_vi = vi;
    endif
  endfor

  iter++;
  eval_hist(iter,:) = cur;

  % fit the model
  fit = M \ vol;
  %fit2 = inv(M'*M)*M'*vol
  %  (fit2 - fit)'

  fit_hist(iter,:) = fit;

  %step = -1*alpha*fit(2:3);
  step = -1*alpha.*fit(2:3);
  if (abs(step(1)) > max_step(1))
    step(1) = sign(step(1))*max_step(1);
  endif
  if (abs(step(2)) > max_step(2))
    step(2) = sign(step(2))*max_step(2);
  endif

endwhile


printf("RSM Search Best Solution: %f %f => %f pL after %d evals\n", 
       best_clk, best_vi, best_vol, num_evals);

figure(2)
plot(eval_hist(:,1), eval_hist(:,2), '*-')
xlabel("Vi")
ylabel("Sample Clock")

figure(3)
subplot(3,1,1)
plot(fit_hist(:,1))
ylabel("Volume (pl)")
subplot(3,1,2)
plot(fit_hist(:,2))
ylabel("dVol / dVi ")
subplot(3,1,3)
plot(fit_hist(:,3))
ylabel("dVol / dClk ")
