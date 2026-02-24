
% Implementazione Adam per tutte le combinazioni di beta1 e beta2

function [x_best_all, f_best_all, history_all] = Adam_multi(x0, dim, bound, n_it, vett_beta1, vett_beta2)

    eta = 1e-3;     % learning rate
    epsilon = 1e-8;     % stabilità numerica
    d = ceil(dim/5); % numero coordinate aggiornate

    n_b1 = length(vett_beta1);
    n_b2 = length(vett_beta2);
    n_exp = n_b1 * n_b2;

    % Output
    x_best_all = zeros(dim, n_exp);
    f_best_all = zeros(1, n_exp);
    history_all = zeros(n_exp, n_it);

    exp_idx = 1;

    for i1 = 1:n_b1
        for i2 = 1:n_b2

            beta1 = vett_beta1(i1);
            beta2 = vett_beta2(i2);

            % Inizializzazione
            x = x0;
            x_best = x;
            f_best = f(x);

            m = zeros(dim,1);
            v = zeros(dim,1);

            for k = 1:n_it

                % Selezione coordinate
                D = randperm(dim, d);

                % Gradiente
                grad = grad_f(x);
                g = grad(D);

                % Momenti
                m(D) = beta1*m(D) + (1-beta1)*g;
                v(D) = beta2*v(D) + (1-beta2)*(g.^2);

                % Bias correction
                m_hat = m(D) ./ (1 - beta1^k);
                v_hat = v(D) ./ (1 - beta2^k);

                % Update
                x(D) = x(D)-eta*m_hat./(sqrt(v_hat)+epsilon);

                % Bound
                x = max(min(x, bound), -bound);

                % Valutazione
                fx = f(x);
                if fx < f_best
                    f_best = fx;
                    x_best = x;
                end

                history_all(exp_idx,k) = f_best;

                fprintf("Adam exp %d (beta1=%.2f, beta2=%.3f) | iter %d | f_best = %.4e\n", ...
                        exp_idx, beta1, beta2, k, f_best);
            end

            % Salvataggio risultati
            x_best_all(:,exp_idx) = x_best;
            f_best_all(exp_idx)   = f_best;

            exp_idx = exp_idx + 1;
        end
    end


    %----------------------------------------------------------------------
    % Plot confronto Adam per diverse combinazioni (beta1, beta2)

    figure; 
    hold on;

    colors     = lines(n_exp);              % colori distinti
    linestyles = {'-','-','-','-'};       % stili ciclici

    exp_idx = 1;
    for i1 = 1:n_b1
        for i2 = 1:n_b2

            style = linestyles{mod(exp_idx-1, length(linestyles)) + 1};

            semilogy(1:n_it, history_all(exp_idx,:), ...
                'Color', colors(exp_idx,:), ...
                'LineStyle', style, ...
                'LineWidth', 1.5, ...
                'DisplayName', sprintf('\\beta_1 = %.2f, \\beta_2 = %.4f', ...
                vett_beta1(i1), vett_beta2(i2)));

            exp_idx = exp_idx + 1;
        end
    end

    xlabel('Iterazione');
    ylabel('Miglior valore f(x)');
    title('Adam: confronto per diverse combinazioni (\beta_1, \beta_2)');
    legend('Location','best');
    grid on;

end
