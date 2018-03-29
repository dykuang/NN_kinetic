function par = set_parameter()
 
 par.R = 8.3144598;
%  par.alpha = (0.1:0.05:0.95)';
 par.alpha = linspace(0.01, 0.99, 51)';
 par.beta = [5, 10, 15, 20, 25];
 

end