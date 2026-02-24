clear all
close all
clc

rng(42); %seed

%--------------------------------------------------------------------------
% Parametri
dim = 10; % Dimensione del problema 
bound = 5.12;
n = 300; % Numero particelle
n_it_PSO = 200; % Numero massimo iterazioni PSO
n_it_CBO = 200; % Numero massimo iterazioni CBO
n_it_grad = 300; % Numero massimo iterazioni SGD
n_it_Adam = 300; % Numero massimo iterazioni Adam
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
%Se la dimensione è 2 visualizzo in un grafico

n_vis = 200;
h=0; %Per PSO
if dim==2

% Griglia f
x1 = linspace(-bound, bound, n_vis);
x2 = linspace(-bound, bound, n_vis);
[X1, X2] = meshgrid(x1, x2);

Z = zeros(size(X1));
for i = 1:size(X1,1)
    for j = 1:size(X1,2)
        Z(i,j) = f([X1(i,j), X2(i,j)]);
    end
end


%--------------------------------------------------------------------------
% Grafico per PSO
figure;
surf(X1, X2, Z);
shading interp;
colorbar;

hold on;
h = scatter3(x(1,:), x(2,:), z, 40, 'r', 'filled');
view(2); %Per vederla dall'alto

xlabel('x_1');
ylabel('x_2');
zlabel('f(x)');
title('Evoluzione della popolazione su f');

axis([-bound bound -bound bound 0 max(Z(:))])
axis manual
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Metodo metaeuristico PSO

disp('Metodo metaeuristico PSO:');

[x_bestPSO, z_bestPSO, historyPSO] = PSO(x,dim,n,bound,n_it_PSO,h);

disp("xbest PSO:"); disp(x_bestPSO);
disp("f(xbest) PSO:"); disp(z_bestPSO);
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%Metodo meaeuritsico CBO

disp('Metodo metaeuristico CBO:');

[x_bestCBO, z_bestCBO, historyCBO] = CBO(x,dim,n,bound,n_it_CBO,h);

disp("xbest CBO:"); disp(x_bestCBO);
disp("f(xbest) CBO:"); disp(z_bestCBO);
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Metodo SGD dopo PSO

disp('Metodo SGD:');

[x_sgd_PSO, f_sgd_PSO, history_sgd_PSO] = RCD_SGD(x_bestPSO, dim, bound, n_it_grad);

disp("xbest SGD:");
disp(x_sgd_PSO);
disp("f(xbest) SGD:");
disp(f_sgd_PSO);
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Metodo SGD dopo CBO

disp('Metodo SGD:');

[x_sgd_CBO, f_sgd_CBO, history_sgd_CBO] = RCD_SGD(x_bestCBO, dim, bound, n_it_grad);

disp("xbest SGD:");
disp(x_sgd_CBO);
disp("f(xbest) SGD:");
disp(f_sgd_CBO);
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Metodo Adam dopo PSO

disp('Metodo Adam:');

[x_Adam_PSO, f_Adam_PSO, history_Adam_PSO] = Adam(x_bestPSO, dim, bound, n_it_Adam);

disp("xbest Adam:");
disp(x_Adam_PSO);
disp("f(xbest) Adam:");
disp(f_Adam_PSO);
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Metodo Adam dopo CBO

disp('Metodo Adam:');

[x_Adam_CBO, f_Adam_CBO, history_Adam_CBO] = Adam(x_bestCBO, dim, bound, n_it_Adam);

disp("xbest Adam:");
disp(x_Adam_CBO);
disp("f(xbest) Adam:");
disp(f_Adam_CBO);
%--------------------------------------------------------------------------



%STAMPA RISULTATI DI TUTTI I METODI IBRIDI
disp("RISULTATI:");

%--------------------------------------------------------------------------
%Miglioramento da metodo metaeuristico (PSO) a basato sul gradiente (SGD)
%Guadagno percentuale

gain_perc_PSO_sgd = 100 * (z_bestPSO - f_sgd_PSO) / abs(z_bestPSO);
fprintf("fbest PSO:"); disp(z_bestPSO);
fprintf("fbest SGD dopo PSO:"); disp(f_sgd_PSO);
fprintf("Guadagno PSO -> SGD: %.4f %%\n", gain_perc_PSO_sgd);
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
%Miglioramento da metodo metaeuristico (CBO) a basato sul gradiente (SGD)
%Guadagno percentuale

gain_perc_CBO_sgd = 100 * (z_bestCBO - f_sgd_CBO) / abs(z_bestCBO);
fprintf("fbest CBO:"); disp(z_bestCBO);
fprintf("fbest SGD dopo CBO:"); disp(f_sgd_CBO);
fprintf("Guadagno CBO -> SGD: %.4f %%\n", gain_perc_CBO_sgd);
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
%Miglioramento da metodo metaeuristico (PSO) a basato sul gradiente (Adam)
%Guadagno percentuale

gain_perc_PSO_Adam = 100 * (z_bestPSO - f_Adam_PSO) / abs(z_bestPSO);
fprintf("fbest PSO:"); disp(z_bestPSO);
fprintf("fbest Adam dopo PSO:"); disp(f_Adam_PSO);
fprintf("Guadagno PSO -> Adam: %.4f %%\n", gain_perc_PSO_Adam);
%--------------------------------------------------------------------------

%-------------------------------------------------------------------------- 
% Grafico comparativo evoluzione f(x) per PSO+SGD, CBO+SGD, PSO+Adam

% Concateno history: metaeuristica + gradiente
evol_PSO_SGD = [historyPSO, history_sgd_PSO];
evol_CBO_SGD = [historyCBO, history_sgd_CBO];
evol_PSO_Adam = [historyPSO, history_Adam_PSO];

% Numero totale di iterazioni per asse x
it_PSO_SGD = 1:length(evol_PSO_SGD);
it_CBO_SGD = 1:length(evol_CBO_SGD);
it_PSO_Adam = 1:length(evol_PSO_Adam);

figure;
semilogy(it_PSO_SGD, evol_PSO_SGD, '-', 'LineWidth', 1.5, 'MarkerIndices', 1:5:length(evol_PSO_SGD));
hold on;
semilogy(it_CBO_SGD, evol_CBO_SGD, '-', 'LineWidth', 1.5, 'MarkerIndices', 1:5:length(evol_CBO_SGD));
hold on;
semilogy(it_PSO_Adam, evol_PSO_Adam, '--', 'LineWidth', 1.5, 'MarkerIndices', 1:5:length(evol_PSO_Adam));


xlabel('Iterazioni');
ylabel('Valore di f(x)');
title('Evoluzione della funzione obiettivo scala logaritmica:');
legend('PSO + SGD', 'CBO + SGD', 'PSO + Adam');
grid on;


figure;
plot(it_PSO_SGD, evol_PSO_SGD, '-', 'LineWidth', 1.5, 'MarkerIndices', 1:5:length(evol_PSO_SGD));
hold on;
plot(it_CBO_SGD, evol_CBO_SGD, '-', 'LineWidth', 1.5, 'MarkerIndices', 1:5:length(evol_CBO_SGD));
hold on;
plot(it_PSO_Adam, evol_PSO_Adam, '--', 'LineWidth', 1.5, 'MarkerIndices', 1:5:length(evol_PSO_Adam));


xlabel('Iterazioni');
ylabel('Valore di f(x)');
title('Evoluzione della funzione obiettivo scala normale:');
legend('PSO + SGD', 'CBO + SGD', 'PSO + Adam');
grid on;
%--------------------------------------------------------------------------
