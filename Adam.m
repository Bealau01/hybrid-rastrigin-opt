%Implementazione algoritmo Adam

function [x_best, f_best, history] = Adam(x, dim, bound, n_it)

    eta = 1e-3;      % learning rate 
    flag = 0;        % flag che indica il motivo della terminazione dell'algoritmo
    n_it_sm = 50;    % numero massimo di iterazioni senza miglioramento
    tol_migl = 1e-6; % tolleranza sul miglioramento
    it_sm = 0;       % contatore
    d=ceil(dim/5);   % dimensione batch sulle coordinate

    %Parametri Adam
    beta1= 0.9;      % controlla quanto i gradienti passati influenzano la media
    beta2=0.999;     % coefficiente di decadimento della media mobile
    epsilon=1e-8;    % per stabilità dell'algoritmo


    %Inizializzazione posizione e valutazione migliore
    x_best = x;
    f_best = f(x);

    %Inizializzazione momento primo e secondo
    m=zeros(dim,1);
    v=zeros(dim,1);

    history = zeros(1,n_it);

    for k = 1:n_it
        
        
        % Selezione casuale di d componenti
        D = randperm(dim);
        D1 = D(1:d);
        
        % Calcolo del gradiente completo
        grad = grad_f(x);
        
        % Aggiornamento solo delle componenti estratte casualmente
        g = grad(D1);
        m(D1) = beta1 * m(D1) + (1 - beta1) * g;
        v(D1) = beta2 * v(D1) + (1 - beta2) * (g.^2);
        
        mc = m(D1) ./ (1 - beta1^k);
        vc = v(D1) ./ (1 - beta2^k);
        
        x(D1) = x(D1) - eta * mc ./ (sqrt(vc + epsilon));

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

    end


% Taglio del vettore history in caso di break anticipato
history = history(1:k-1);

% Plot andamento miglior valore
    figure;
    semilogy(1:length(history), history, '-', 'LineWidth', 1.5);
    xlabel('Iterazione');
    ylabel('Miglior valore f(y_{best})');
    title(sprintf('Andamento del miglior valore Adam (f_{best} = %.6e)', f_best));
    grid on;

%Stampa motivo terminazione dell'algoritmo
switch flag 
    case 0
        fprintf("Adam terminato per numero massimo di iterazioni raggiunte \n");
    case 1
        fprintf("Adam terminato per numero massimo di iterazioni senza miglioramento \n");
end
    
end


