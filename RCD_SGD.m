%Metodo SGD che prevede calcolo del gradiente solo in un sottoinsieme casuale di direzioni
%Random Coordinate Descent 

function [x_best, f_best, history] = RCD_SGD(x, dim, bound, n_it)

    eta0 = 1e-2;     % learning rate 
    flag = 0;        % flag che indica il motivo della terminazione dell'algoritmo
    n_it_sm = 50;    % numero massimo di iterazioni senza miglioramento
    tol_migl = 1e-6;% tolleranza sul miglioramento
    it_sm = 0;       % contatore
    d=ceil(dim/5);   % dimensione batch sulle coordinate

    %Inizializzazione posizione e valutazione migliore
    x_best = x;
    f_best = f(x);

    history = zeros(1,n_it);

    for k = 1:n_it
        
        eta = eta0/k;   % learning rate decrescente - Robbins-monro

        % Selezione casuale di d componenti
        D = randperm(dim);
        D1 = D(1:d);
        
        % Calcolo del gradiente completo
        grad = grad_f(x);
        
        % Aggiornamento solo delle componenti estratte casualmente
        for i=1:d
            d1=D1(i);
            x(d1) = x(d1) - eta * grad(d1);
        end 

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
    title(sprintf('Andamento del miglior valore RCD-SGD (f_{best} = %.6e)', f_best));
    grid on;

%Stampa motivo terminazione dell'algoritmo
switch flag 
    case 0
        fprintf("SGD terminato per numero massimo di iterazioni raggiunte \n");
    case 1
        fprintf("SGD terminato per numero massimo di iterazioni senza miglioramento \n");
end
    
end
