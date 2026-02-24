% Implementazione metodo CBO

% A partire da una popolazione, trovare la soluzione finale migliore

function [y_best, f_y_best, history] = CBO(x,dim,n,bound,n_it,h)

    flag = 0; % motivo terminazione

    % Parametri CBO
    alpha  = 100;      % intensità selezione - peso di Gibbs: quanto peso si vuole dare alle particelle migliori
    lambda = 0.2;     % drift verso consenso (piccolo-> più tempo per raggiungere il consenso)
    sigma  = 2;     % intensità rumore - esplorazione
    dt     = 0.1;     % passo temporale

    % Valutazioni iniziali
    z = zeros(1,n);
    for i = 1:n
        z(i) = f(x(:,i));
    end

    [f_y_best, ~] = min(z);
    %y_best = x(:,i_best);

    % Parametri di stop
    tol_x = 1e-10;       % convergenza popolazione
    n_it_sm = 40;       % max iterazioni senza miglioramento
    tol_migl = 1e-10;
    it_sm = 0;

    % Inizializzazione storia miglior valore
    history = zeros(1,n_it);


    % Ciclo principale
    k = 1;
    while k <= n_it

        f_current_best = f_y_best;


    % Consensus Point 
    z_min = min(z);
    % Sottraendo z_min, l'esponente della particella migliore sarà 0
    % exp(0)=1, quindi avrà sempre almeno un peso non nullo.
    weights = exp(-alpha * (z - z_min)); 
    
    % Aggiungi un epsilon per sicurezza estrema contro la divisione per zero
    weights = weights / (sum(weights) + 1e-30); 
    
    m = zeros(dim,1);
    for i = 1:n
        m = m + weights(i) * x(:,i);
    end
    % ---------------------------------------

    
    %{
        % Consensus point
        weights = exp(-alpha * z);
        weights = weights / sum(weights); %distribuzione di Gibbs

        m = zeros(dim,1);
        for i = 1:n
            m = m + weights(i) * x(:,i);
        end
%}


        % Aggiornamento particelle 
        for i = 1:n 
            noise = randn(dim,1); % vettore rumore normalizzato 
            x(:,i) = x(:,i) - lambda*(x(:,i)-m)*dt + sigma*abs(x(:,i)-m).*noise*sqrt(dt); %caso anisotropo 
            % valore assoluto per implementazione più robusta 
        end

        % Rispetto dei bounds
        x = max(min(x, bound), -bound);

        % Valutazione funzione nei nuovi punti
        for i = 1:n
            z(i) = f(x(:,i));
        end

        [f_y_best, i_best] = min(z);
        y_best = x(:,i_best);

        history(k) = f_y_best;

        
        if f_y_best < min(history(1:k-1))
            x_best = y_best;  % aggiorna la posizione del minimo assoluto
        end

        % Stop su miglioramento
        if f_y_best < f_current_best*(1 - tol_migl)
            it_sm = 0;
        else
            it_sm = it_sm + 1;
        end
        if it_sm >= n_it_sm
            flag = 2;
            break
        end

        % Collasso popolazione: dispersione vicina a zero
        spread = max(vecnorm(x-m, 2, 1));
        if spread < tol_x
            flag = 1;
            break
        end

        % Grafico se dim=2
        if dim == 2
            set(h, 'XData', x(1,:), 'YData', x(2,:), 'ZData', z);
            drawnow;    

            pause(0.1);
        end

        fprintf("CBO f_best all'iterazione %d: %.4e \n", k, f_y_best);

        k = k + 1;
    end

    % Taglio history
    history = history(1:k-1);



    f_y_best=min(history);
    y_best=x_best;

    % Plot andamento miglior valore
    figure;
    plot(1:length(history), history, '-', 'LineWidth', 1.5);
    xlabel('Iterazione');
    ylabel('Miglior valore f(x)');
    title(sprintf('Andamento del miglior valore CBO (f_{best} = %.6e)', f_y_best));
    grid on;

    % Motivo di terminazione CBO
    switch flag
        case 0
            disp("CBO terminato per numero massimo di iterazioni raggiunte");
        case 1
            disp("CBO terminato per collasso della popolazione (consenso)");
        case 2
            disp("CBO terminato per numero massimo di iterazioni senza miglioramento");
    end

end
