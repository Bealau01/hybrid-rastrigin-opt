
clear all
close all
clc

rng(42); %seed

%--------------------------------------------------------------------------
% Parametri
dim = 2; % Dimensione del problema 
bound = 5.12;
n = 50; % Numero particelle
n_it_PSO = 100; % Numero massimo iterazioni PSO
n_it_CBO = 100; % Numero massimo iterazioni CBO
n_it_grad = 200; % Numero massimo iterazioni SGD
n_it_Adam = 200; % Numero massimo iterazioni Adam
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Inizializzazione della popolazione
x = -bound + 2*bound*rand(dim,n);
z = zeros(1,n);
for i = 1:n
    z(i) = f(x(:,i));
end
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Metodo metaeuristico PSO con diversi w
vett_w = [0.4,0.7,1];

disp('Metodo metaeuristico PSO variando w:');

[x_bestPSO, f_bestPSO, historyPSO] = PSO_multiw(x,dim,n,bound,n_it_PSO,vett_w);

for j = 1:length(vett_w)
    fprintf("w = %.2f -> f_best = %.6e\n", vett_w(j), f_bestPSO(j));
end


%Grafico confronto PSO per diversi w
figure;
plot(historyPSO.', 'LineWidth', 1.5);   
xlabel('Iterazione');
ylabel('Miglior valore f(y_{best})');
legend(arrayfun(@(w) sprintf('w = %.2f', w), vett_w, 'UniformOutput', false));
grid on;
title('Confronto PSO per diversi valori di w');
%--------------------------------------------------------------------------



%-------------------------------------------------------------------------- 
% Metodo metaeuristico CBO per tutte le combinazioni di lambda e sigma

vett_lambda = [0.2, 0.5];
vett_sigma  = [1, 2];

disp('Metodo metaeuristico CBO per tutte le combinazioni di lambda e sigma:');

[y_bestCBO, f_bestCBO, historyCBO] = CBO_multi(x, dim, n, bound, n_it_CBO, vett_lambda, vett_sigma);

n_exp = length(f_bestCBO);

% Stampa risultati
exp_idx = 1;
for li = 1:length(vett_lambda)
    for si = 1:length(vett_sigma)
        fprintf("lambda = %.2f, sigma = %.2f -> f_best = %.6e\n", ...
                vett_lambda(li), vett_sigma(si), f_bestCBO(exp_idx));
        exp_idx = exp_idx + 1;
    end
end

% Grafico confronto CBO per tutte le combinazioni
figure;
semilogy(historyCBO.', 'LineWidth', 1.5);
xlabel('Iterazione');
ylabel('Miglior valore f(y_{best})');

% Creazione legende
legende = cell(1, n_exp);
exp_idx = 1;
for li = 1:length(vett_lambda)
    for si = 1:length(vett_sigma)
        legende{exp_idx} = sprintf('\\lambda = %.2f, \\sigma = %.2f', vett_lambda(li), vett_sigma(si));
        exp_idx = exp_idx + 1;
    end
end

legend(legende);
grid on;
title('Confronto CBO per diverse combinazioni di lambda e sigma');
%--------------------------------------------------------------------------





%--------------------------------------------------------------------------
% Metodo SGD dopo PSO: (w, eta)

vett_eta = [1e-2, 1e-3, 1e-4];
n_w   = length(vett_w);
n_eta = length(vett_eta);

disp('Metodo SGD dopo PSO (per ogni w e ogni eta):');

% Celle per salvare le history
history_sgd_PSO = cell(n_w, n_eta);

% Matrici risultati finali
x_sgd_PSO = zeros(dim, n_w, n_eta);
f_sgd_PSO = zeros(n_w, n_eta);

for j = 1:n_w
    fprintf('\nSGD a partire da PSO con w = %.2f\n', vett_w(j));

    % Punto iniziale = best PSO con quel w
    x0 = x_bestPSO(:,j);

    % SGD multi-eta
    [x_best_sgd, f_best_sgd, history_sgd] = RCD_SGD_multieta(x0, dim, bound, n_it_grad, vett_eta);

    % Salvataggio risultati
    for k = 1:n_eta
        x_sgd_PSO(:,j,k)   = x_best_sgd(:,k);
        f_sgd_PSO(j,k)     = f_best_sgd(k);
        history_sgd_PSO{j,k} = history_sgd(k,:);
        
        fprintf("  eta = %.1e -> f_best = %.6e\n", vett_eta(k), f_best_sgd(k));
    end
end


% Grafico PSO + SGD concatenati 

figure;

for j = 1:n_w
    for k = 1:n_eta
        
        % History PSO per w_j
        h_pso = historyPSO(j,:);
        
        % History SGD per (w_j, eta_k)
        h_sgd = history_sgd_PSO{j,k};
        
        % Concatenazione
        h_tot = [h_pso, h_sgd];
        
        % Plot curva completa
        semilogy(h_tot, 'LineWidth', 1.3, 'DisplayName', sprintf('w = %.2f, \\eta = %.1e', vett_w(j), vett_eta(k)));
         hold on;
    end
end

xlabel('Iterazione');
%ylabel('Miglior valore f(x)');
title('Strategie ibride PSO–SGD per diverse coppie (w, \eta)');
legend('Location', 'best', 'Orientation', 'horizontal', 'NumColumns', 2);
grid on;
%--------------------------------------------------------------------------


%-------------------------------------------------------------------------- 
% Metodo ibrido CBO + SGD: tutte le combinazioni lambda-sigma + eta

% Numero combinazioni
n_lambda = length(vett_lambda);
n_sigma  = length(vett_sigma);

disp('Metodo SGD dopo CBO per ogni combinazione (lambda, sigma) e ogni eta:');

% Celles per salvare le history di SGD
history_sgd_CBO = cell(n_lambda, n_sigma, n_eta);

% Matrici risultati finali
x_sgd_CBO = zeros(dim, n_lambda, n_sigma, n_eta);
f_sgd_CBO = zeros(n_lambda, n_sigma, n_eta);

% Loop sulle combinazioni di CBO
exp_idx = 1;
for li = 1:n_lambda
    for si = 1:n_sigma
        
        % Punto iniziale = miglior CBO per quella combinazione
        x0 = y_bestCBO(:,exp_idx);
        f0 = f_bestCBO(exp_idx);
        
        fprintf('\nSGD a partire da CBO con lambda = %.2f, sigma = %.2f\n', vett_lambda(li), vett_sigma(si));
        
        % SGD multi-eta
        [x_best_sgd, f_best_sgd, history_sgd] = RCD_SGD_multieta(x0, dim, bound, n_it_grad, vett_eta);
        
        % Salvataggio risultati
        for k = 1:n_eta
            x_sgd_CBO(:,li,si,k)   = x_best_sgd(:,k);
            f_sgd_CBO(li,si,k)     = f_best_sgd(k);
            history_sgd_CBO{li,si,k} = history_sgd(k,:);
            
            fprintf("  eta = %.1e -> f_best = %.6e\n", vett_eta(k), f_best_sgd(k));
        end
        
        exp_idx = exp_idx + 1;
    end
end

% Grafico CBO + SGD concatenati

figure;

% Palette colori e stili per differenziare bene le curve
colors = lines(n_lambda*n_sigma);   % palette MATLAB con molti colori
linestyles = {'-','--',':','-.'};  % cicla tra stili di linea
color_idx = 1;

exp_idx = 1;
for li = 1:n_lambda
    for si = 1:n_sigma
        for k = 1:n_eta
            % History CBO
            h_cbo = historyCBO(exp_idx,:);
            
            % History SGD
            h_sgd = history_sgd_CBO{li,si,k};
            
            % Concatenazione
            h_tot = [h_cbo, h_sgd];
            
            % Scelta colore e stile
            clr = colors(color_idx,:);
            style = linestyles{mod(k-1,length(linestyles))+1}; % cicla sugli stili
            
            % Plot
            semilogy(h_tot, 'Color', clr, 'LineStyle', style, 'LineWidth', 1.3, ...
                     'DisplayName', sprintf('\\lambda = %.2f, \\sigma = %.2f, \\eta = %.1e', vett_lambda(li), vett_sigma(si), vett_eta(k)));
            
            hold on;
        end
        color_idx = color_idx + 1;
        exp_idx = exp_idx + 1;
    end
end

xlabel('Iterazione');
%ylabel('Miglior valore f(x)');
title('Strategie ibride CBO–SGD per combinazioni (\lambda, \sigma, \eta)');
legend('Location', 'best', 'Orientation', 'horizontal', 'NumColumns', 2);
grid on;
%--------------------------------------------------------------------------


%-------------------------------------------------------------------------- 
% Metodo ibrido PSO + Adam: tutte le combinazioni w + beta1 , beta2

vett_beta1 = [0.5, 0.9];
vett_beta2 = [0.8, 0.999];

n_b1 = length(vett_beta1);
n_b2 = length(vett_beta2);
n_exp_adam = n_b1 * n_b2;

% Celle history Adam
history_adam_PSO = cell(n_w, n_exp_adam);

% Risultati finali
x_adam_PSO = zeros(dim, n_w, n_exp_adam);
f_adam_PSO = zeros(n_w, n_exp_adam);

disp('Metodo Adam dopo PSO per tutte le combinazioni (w, beta1, beta2):');

for j = 1:n_w
    fprintf('\nAdam a partire da PSO con w = %.2f\n', vett_w(j));

    % Punto iniziale = best PSO
    x0 = x_bestPSO(:,j);

    % Unica chiamata
    [x_best_all, f_best_all, history_all] = Adam_multi( ...
        x0, dim, bound, n_it_grad, vett_beta1, vett_beta2);

    % Salvataggio risultati
    for e = 1:n_exp_adam
        x_adam_PSO(:,j,e)   = x_best_all(:,e);
        f_adam_PSO(j,e)     = f_best_all(e);
        history_adam_PSO{j,e} = history_all(e,:);

        fprintf("  exp %d -> f_best = %.6e\n", e, f_best_all(e));
    end
end


%-------------------------------------------------------------------------- 
% Grafico PSO + Adam concatenati

figure;
colors     = lines(n_w);                 % un colore per ogni w
linestyles = {'-','--',':','-.'};        

for j = 1:n_w
    exp_idx = 1;
    for b1 = 1:n_b1
        for b2 = 1:n_b2

            % History PSO
            h_pso = historyPSO(j,:);

            % History Adam
            h_adam = history_adam_PSO{j,exp_idx};

            % Concatenazione
            h_tot = [h_pso, h_adam];

            semilogy(h_tot, ...
                'Color', colors(j,:), ...
                'LineStyle', linestyles{mod(exp_idx-1, 4) + 1}, ...
                'LineWidth', 1.3, ...
                'DisplayName', sprintf( ...
                'w=%.2f, \\beta_1=%.2f, \\beta_2=%.4f', ...
                vett_w(j), vett_beta1(b1), vett_beta2(b2)));

            hold on;
            exp_idx = exp_idx + 1;
        end
    end
end

xlabel('Iterazione');
%ylabel('Miglior valore f(x)');
title('Strategie ibride PSO-Adam per tutte le combinazioni (w,\beta_1,\beta_2)');
legend('Location', 'best', 'Orientation', 'horizontal', 'NumColumns', 2);
grid on;


