## Code originally written by Vladimir Sobes, adapted by Felipe Flores

function [x, data_1,data_2,true_value_1,true_value_2,vector_of_poles,vector_of_residues] = generate_data(number_of_poles,relative_width_of_poles,number_points_per_pole,signal_to_noise_ratio)
  
%% Data control parameters
% number_of_poles = 1e1;
% relative_width_of_poles = 1e-2; % ratio of average pole width to average level spacing 
% number_points_per_pole = 1e2;  
% signal_to_noise_ratio = 1e-3;

  %% Average Resonance Parameters
  
  average_level_spacing  = 2/(number_of_poles/2);
  average_neutron_width = 2/3*relative_width_of_poles*average_level_spacing;
  average_capture_width = average_neutron_width/2;
  degrees_of_freedom_for_capture_channels   = 9e2*rand(1) + 1e2; % rand(1) is a random number from a uniform distribution [0,1]
  
  %% Assign poles
  
  %vector_of_poles = zeros(0,2); % preallocate a vector to be expanded
  vector_of_poles = [];
  vector_of_residues = zeros(0,2); % preallocate a vector to be expanded
  
  while length(vector_of_poles) < number_of_poles % populate pole list until desired number of poles is reached               
      m = zeros(2); % preallocate a 2x2 matrix of zeros
      m(1,1) = randn(1); % randn(1) samples a standard normal distribution mean 0, variance 1
      m(2,2) = randn(1);
      m(1,2) = sqrt(1/2)*randn(1); 
      m(2,1) = m(1,2);
      s = sort(eig(m)); % eig(m) returns the 2 eigenvalues of m.  sort() returns s(1) < s(2)
      d = average_level_spacing*(s(2) - s(1))/2;
       
      % place the first pole randomly if 1) we don't have any poles, 2) this
      % pole would go beyond +1
      if size(vector_of_poles,1) == 0 || vector_of_poles(end) + d > 1 
         dPole = -1 + average_level_spacing*rand(1);
      else        
         dPole = vector_of_poles(end) + d;        
      end
      
      vector_of_poles = [vector_of_poles; dPole]; % append pole list
      
      Gn = average_neutron_width*chi2rnd(1); % chi2rnd(1) samples a chi-squared distribution with 1 degree of freedom
      Gg = average_capture_width*chi2rnd(degrees_of_freedom_for_capture_channels)/degrees_of_freedom_for_capture_channels; % chi2rnd(x) samples a chi-squared distribution with x degrees of freedom
      Gt = Gn + Gg;       
  
      vector_of_poles(end) = vector_of_poles(end) + 1i*Gt; 
      vector_of_residues = [vector_of_residues; Gn + -1i*Gn*Gg/Gt, -1i*Gn*Gg/Gt]; % append residue list
  end
  
  number_of_poles = length(vector_of_poles);        
  
  %% 
  
  number_of_data_points = number_points_per_pole*number_of_poles;
  x = linspace(-1,1,number_of_data_points)';
  
  C = 1e0; % additive constant manually chosen to give positive true values
  scale = 1/signal_to_noise_ratio^2; % rescale the true values to generate poisson noise
  
  calculate_true_value = @(residue,pole,x) real(calculate_Cauchy_matrix(x,pole)*residue); % in-line function definition
  
  % linear scaling of the true values
  true_value_1 = scale*(calculate_true_value(vector_of_residues(:,1),vector_of_poles,x) + C);
  true_value_2 = scale*(calculate_true_value(vector_of_residues(:,2),vector_of_poles,x) + C);
  
  %% Noise model
  
  data_1 = zeros(size(true_value_1));
  data_2 = zeros(size(true_value_2));
  for ix = 1:number_of_data_points
      data_1(ix) = poissrnd(true_value_1(ix)); % poissrnd(x) draws a random value from a poisson discrete distribution with mean x
      data_2(ix) = poissrnd(true_value_2(ix));
  end
  
  % rescale back
  true_value_1 = true_value_1/scale;
  true_value_2 = true_value_2/scale;
  
  data_1 = data_1/scale;
  data_2 = data_2/scale;
