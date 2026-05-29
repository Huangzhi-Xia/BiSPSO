%% Particle Swarm Optimization with Bidirectional Search (BiSPSO)
function [Best_pos, Best_score, curve] = BiSPSO(pop_size, Max_iter, lb, ub, dim, fobj, Vmax, Vmin)
% Input parameters:
%   pop_size : Population size
%   Max_iter : Maximum number of iterations
%   lb, ub   : Lower and upper bounds
%   dim      : Dimension of the problem
%   fobj     : Objective function handle
%   Vmax, Vmin: Maximum and minimum velocity bounds. Please set them to 2 and -2 respectively.

    %% Parameter initialization
    w_max = 0.9; w_min = 0.1;
    c1_initial = 1.0; c1_final = 0.5;
    c2_initial = 0.5; c2_final = 1.0;  
    
    % Unify the size of bounds and velocity limits matrices
    if(max(size(ub)) == 1)
        ub = ub .* ones(1, dim);
        lb = lb .* ones(1, dim);
    end
    if(max(size(Vmax)) == 1)
        Vmax = Vmax .* ones(1, dim);
        Vmin = Vmin .* ones(1, dim);
    end  
    
    %% Population initialization
    Range = ones(pop_size, 1) * (ub - lb);
    population = rand(pop_size, dim) .* Range + ones(pop_size, 1) * lb; % Randomly initialize position
    velocity = rand(pop_size, dim) .* (ones(pop_size, 1) * (Vmax - Vmin)) + ones(pop_size, 1) * Vmin; % Initialize velocity  
    
    personal_best_scores = zeros(pop_size, 1);
    for i = 1:pop_size
        personal_best_scores(i) = fobj(population(i, :)); % Calculate initial fitness values
    end  
    
    % Initialize personal best and global best
    personal_best = population;
    [best_score, best_idx] = min(personal_best_scores);
    global_best = personal_best(best_idx, :);  
    
    curve = zeros(1, Max_iter);  
    
    %% Main loop
    for iter = 1:Max_iter
        % Adaptive parameter update for inertia weight and cognitive/social terms
        w = w_max - (w_max - w_min) * iter / Max_iter;
        c1 = c1_initial - (c1_initial - c1_final) * iter / Max_iter;
        c2 = c2_initial + (c2_final - c2_initial) * iter / Max_iter;  
        
        % Calculate population diversity
        center = mean(population);
        diversity = sum(sqrt(sum((population - ones(pop_size, 1) * center).^2, 2))) / pop_size;
        
        f_worst = max(personal_best_scores);
        
        for i = 1:pop_size
            % Adaptive inertia weight adjustment based on fitness
            fitness_ratio = (personal_best_scores(i) - best_score) / (f_worst - best_score + 1e-10);
            adaptive_w = w * (1 + 0.5 * fitness_ratio);  
            
            r1 = rand(1, dim);
            r2 = rand(1, dim);  
            
            % Update velocity with cognitive and social terms
            velocity(i, :) = adaptive_w * velocity(i, :) + ...
                c1 * r1 .* (personal_best(i, :) - population(i, :)) + ...
                c2 * r2 .* (global_best - population(i, :));  
            
            % Adaptive maximum velocity limitation
            current_vmax = Vmax .* (1 - 0.5 * (iter / Max_iter));
            current_vmin = Vmin .* (1 - 0.5 * (iter / Max_iter));
            velocity(i, :) = max(current_vmin, min(current_vmax, velocity(i, :)));  
            
            % Update position
            population(i, :) = population(i, :) + velocity(i, :);  
            
            % Perform boundary constraint
            population(i, :) = max(lb, min(ub, population(i, :)));  
            
            % Evaluate the fitness value
            fitness = fobj(population(i, :));  
            
            % Update personal best
            if fitness < personal_best_scores(i)
                personal_best_scores(i) = fitness;
                personal_best(i, :) = population(i, :);  
                
                % Update global best
                if fitness < best_score
                    best_score = fitness;
                    global_best = population(i, :);
                    best_idx = i;
                end
            end
        end  

        %% Bidirectional Search Strategy
        % Perform bidirectional search periodically to enhance local exploitation
        if mod(iter, 20) == 0
            T_local = 5;
            eta_0 = 0.05;
            
            for i = 1:pop_size
                curr_x = population(i, :);
                curr_f = fobj(curr_x);
                
                for iter_l = 1:T_local
                    % Adaptively calculate the search step size
                    eta = eta_0 * (0.9 ^ iter_l);
                    step = eta .* (ub - lb);
                    
                    % Positive direction search and boundary constraint
                    x_pos = max(lb, min(ub, curr_x + step));
                    f_pos = fobj(x_pos);
                    
                    % Negative direction search and boundary constraint
                    x_neg = max(lb, min(ub, curr_x - step));
                    f_neg = fobj(x_neg);
                    
                    % Select particles with better fitness values through the greedy strategy
                    if f_pos < curr_f && f_pos <= f_neg
                        curr_x = x_pos;
                        curr_f = f_pos;
                    elseif f_neg < curr_f && f_neg < f_pos
                        curr_x = x_neg;
                        curr_f = f_neg;
                    end
                end
                
                % Update particle information after bidirectional search
                population(i, :) = curr_x;
                if curr_f < personal_best_scores(i)
                    personal_best_scores(i) = curr_f;
                    personal_best(i, :) = curr_x;
                    if curr_f < best_score
                        best_score = curr_f;
                        global_best = curr_x;
                    end
                end
            end
        end
        
        %% Reinitialization Strategy
        % Reinitialize particles when population diversity is too low to avoid premature convergence
        if diversity < 0.01 && iter < Max_iter * 0.8
            % Find the particles with the poorest performance for reinitialization
            worst_indices = find(personal_best_scores > prctile(personal_best_scores, 80));
            for idx = worst_indices'
                % Protect the global best particle from being reinitialized
                if idx ~= best_idx 
                    % Reinitialize the position and velocity
                    population(idx, :) = rand(1, dim) .* (ub - lb) + lb;
                    velocity(idx, :) = 0.1 * (rand(1, dim) - 0.5) .* (ub - lb);
                    
                    % Update fitness value and personal best
                    personal_best_scores(idx) = fobj(population(idx, :));
                    personal_best(idx, :) = population(idx, :);
                end
            end
        end  
        
        curve(iter) = best_score;
    end  
    
    %% Return the best solution
    Best_pos = global_best;
    Best_score = best_score;
end