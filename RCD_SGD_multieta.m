
% Random Coordinate Descent - SGD per diversi learning rate eta


function [x_best, f_best, history] = RCD_SGD_multieta(x0, dim, bound, n_it, vett_eta)

    n_eta = length(vett_eta);
    d = ceil(dim/5);          % numero di coordinate aggiornate

    % Output
    x_best = zeros(dim, n_eta);
    f_best = zeros(1, n_eta);
    history = zeros(n_eta, n_it);

    % Ciclo sui diversi learning rate
    for j = 1:n_eta

        eta0 = vett_eta(j);
        x = x0;

        % Inizializzazione
        x_best_j = x;
        f_best_j = f(x);

        for k = 1:n_it

            % Robbins–Monro
            eta = eta0 / k;

            % Selezione casuale coordinate
            D = randperm(dim, d);

            % Gradiente completo
            grad = grad_f(x);

            % Aggiornamento solo sulle coordinate estratte
            for i = 1:d
                idx = D(i);
                x(idx) = x(idx) - eta * grad(idx);
            end

            % Rispetto dei bound
            x = max(min(x, bound), -bound);

            % Valutazione
            fx = f(x);

            % Aggiornamento miglior valore
            if fx < f_best_j
                f_best_j = fx;
                x_best_j = x;
            end

            history(j,k) = f_best_j;
        end

        % Salvataggio risultati
        x_best(:,j) = x_best_j;
        f_best(j)   = f_best_j;

        fprintf("RCD-SGD terminato (eta = %.2e), f_best = %.6e \n", ...
                eta0, f_best_j);
    end

    % Plot confronto learning rate
    figure; 
    hold on;
    for j = 1:n_eta
        semilogy(1:n_it, history(j,:), 'LineWidth', 1.5);
    end
    xlabel('Iterazione');
    ylabel('Miglior valore f(x)');
    legend(arrayfun(@(e) sprintf('\\eta = %.1e', e), vett_eta, ...
           'UniformOutput', false));
    title('RCD-SGD: confronto per diversi learning rate');
    grid on;
end
