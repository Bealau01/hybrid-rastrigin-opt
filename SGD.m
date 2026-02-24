%Metodo SGD che prevede GD calcolato numericamente con il metodo delle
%differenze finite centrali con l'intoroduzione di un termine stocastico
%dato da un rumore nel gradiente

function [x_best, f_best, history] = SGD(x, dim, bound, n_it)

    eta0 = 1e-3;     % learning rate
    tol_grad = 1e-6; % criterio di stop
    noise = 1e-4;    % rumore per rendere il GD stocastico
    flag = 0;        % flag che indica il motivo della terminazione dell'algoritmo
    n_it_sm = 30;    % numero massimo di iterazioni senza miglioramento
    tol_migl = 1e-10;% tolleranza sul miglioramento
    it_sm = 0;       % contatore

    %Inizializzazione posizione e valutazione migliore
    x_best = x;
    f_best = f(x);

    history = zeros(1,n_it);

    for k = 1:n_it
        
        eta = eta0/k;   % learning rate decrescente

        % Gradiente esatto con aggiunta di rumore
        grad = grad_f(x) + noise * randn(dim,1);

        % Aggiornamento SGD
        x = x - eta * grad;

        % Controllo bound
        x = max(min(x, bound), -bound);


        % Valutazione funzione su x attuale
        fx = f(x);
        


        % Stop sul miglioramento
        if fx < f_best*(1-tol_migl)
            f_best = fx;
            x_best = x;
            it_sm = 0; % reset contatore
        else
            it_sm = it_sm + 1;
        end

        fprintf("f e f_best all'iterazione %d: %.6e - %.6e \n", k, fx,f_best);
        history(k)=f_best;

        if it_sm >= n_it_sm
            flag = 1;
            break
        end

        % Stop su gradiente
        if norm(grad) < tol_grad
            flag =2;
            break
        end

    end

% Taglio del vettore history in caso di break anticipato
history = history(1:k-1);

% Plot andamento miglior valore
    figure;
    semilogy(1:length(history), history, '-', 'LineWidth', 1.5);
    xlabel('Iterazione');
    ylabel('Miglior valore f(y_{best})');
    title(sprintf('Andamento del miglior valore SGD (f_{best} = %.6e)', f_best));
    grid on;


%Stampa motivo terminazione dell'algoritmo
switch flag 
    case 0
        fprintf("SGD terminato per numero massimo di iterazioni raggiunte \n");
    case 1
        fprintf("SGD terminato per numero massimo di iterazioni senza miglioramento \n");
    case 2
        fprintf("SGD terminato per gradiente ~ 0 (iter %d)\n", k);
end
    
end
