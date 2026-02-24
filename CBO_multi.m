
% Implementazione CBO per tutte le combinazioni di lambda e sigma

function [y_best_all, f_y_best_all, history_all] = CBO_multi(x, dim, n, bound, n_it, vett_lambda, vett_sigma)

    alpha = 100;  % intensità selezione
    dt    = 0.1;  % passo temporale

    n_lambda = length(vett_lambda);
    n_sigma  = length(vett_sigma);
    n_exp    = n_lambda * n_sigma;  % numero totale combinazioni

    % Inizializzazione output
    y_best_all   = zeros(dim, n_exp);
    f_y_best_all = zeros(1, n_exp);
    history_all  = zeros(n_exp, n_it);

    exp_idx = 1; % contatore combinazioni
    for li = 1:n_lambda
        for si = 1:n_sigma

            lambda = vett_lambda(li);
            sigma  = vett_sigma(si);

            % Copia stato iniziale
            xk = x;

            % Valutazioni iniziali
            z = zeros(1,n);
            for i = 1:n
                z(i) = f(xk(:,i));
            end

            [f_best, idx] = min(z);
            x_best = xk(:,idx);

            % Ciclo CBO
            for k = 1:n_it

                % Consensus point
                z_min   = min(z);
                weights = exp(-alpha*(z - z_min));
                weights = weights / (sum(weights) + 1e-30);

                m = zeros(dim,1);
                for i = 1:n
                    m = m + weights(i)*xk(:,i);
                end

                % Aggiornamento particelle
                for i = 1:n
                    noise = randn(dim,1);
                    xk(:,i) = xk(:,i) - lambda*(xk(:,i)-m)*dt + sigma*abs(xk(:,i)-m).*noise*sqrt(dt);
                end

                % Controllo bound
                xk = max(min(xk, bound), -bound);

                % Nuove valutazioni
                for i = 1:n
                    z(i) = f(xk(:,i));
                end

                [f_curr, idx] = min(z);
                if f_curr < f_best
                    f_best = f_curr;
                    x_best = xk(:,idx);
                end

                history_all(exp_idx,k) = f_best;

                fprintf("CBO exp %d (lambda=%.2f, sigma=%.2f) | iter %d | f_best = %.4e\n", ...
                        exp_idx, lambda, sigma, k, f_best);
            end

            % Salvataggio risultati 
            y_best_all(:,exp_idx) = x_best;
            f_y_best_all(exp_idx) = f_best;

            exp_idx = exp_idx + 1;
        end
    end
end
