
% Implementazione PSO per diversi valori del parametro di inerzia w

function [y_best_all, f_y_best_all, history] = PSO_multiw(x,dim,n,bound,n_it_PSO,vett_w)

n_w = length(vett_w);

% Preallocazioni
history = zeros(n_w, n_it_PSO);
y_best_all = zeros(dim, n_w);
f_y_best_all = zeros(1, n_w);

% Parametri PSO fissi
c1 = 1.5;
c2 = 1.5;

for j = 1:n_w

    % Inerzia corrente
    w = vett_w(j);

    % Stesse condizioni iniziali per confronto equo
    xj = x;
    y  = xj;

    % Inizializzazione pbest
    f_y = zeros(1,n);
    for i = 1:n
        f_y(i) = f(y(:,i));
    end

    [~, i_best] = min(f_y);
    y_best = y(:,i_best);

    % Velocità iniziali
    v = rand(dim,n);

    % Ciclo PSO
    for k = 1:n_it_PSO

        % Valutazione fitness e aggiornamento pbest / gbest
        for i = 1:n
            fi = f(xj(:,i));
            if fi < f(y(:,i))
                y(:,i) = xj(:,i);
            end
            if f(y(:,i)) < f(y_best)
                y_best = y(:,i);
            end
        end

        % Aggiornamento velocità e posizioni
        for i = 1:n
            r1 = rand(dim,1);
            r2 = rand(dim,1);
            v(:,i) = w*v(:,i) ...
                   + c1*r1.*(y(:,i) - xj(:,i)) ...
                   + c2*r2.*(y_best - xj(:,i));
            xj(:,i) = xj(:,i) + v(:,i);
        end

        % Rispetto dei bounds
        xj = max(min(xj, bound), -bound);

        % Salvataggio storico
        history(j,k) = f(y_best);

        fprintf("w = %.2f | iter %d | f_best = %.6e\n", ...
                w, k, history(j,k));
    end

    % Salvataggio risultati finali per questo w
    y_best_all(:,j) = y_best;
    f_y_best_all(j) = f(y_best);

end

end
