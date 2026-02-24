%Implementazione metodo PSO

% A partire da una popolazione, trovare la soluzione finale migliore

function [y_best, f_y_best, history] = PSO(x,dim,n,bound,n_it_PSO,h)

        y = x; %Inizializzazione delle miglior posizioni locali 
        flag = 0; %Motivo terminazione

        %Inizializzazione miglior posizione globale y_best (e suo valore f_y_best)
        f_y = zeros(1,n);
        for i=1:n
            f_y(i) = f(y(:,i));
        end
        [f_y_best,i_best]=min(f_y);
        y_best=y(:,i_best);

        v = rand(dim,n); %Inizializzazione delle velocità

        %Inizializzazione parametri
        w  = 0.7; %inerzia costante
        c1 = 1.5;
        c2 = 1.5;

        z = zeros(1,n); %Inizializzazione vettore valutazione fitness


        % Contatore per iterazioni senza miglioramento
        n_it_sm = 40;
        tol_migl = 1e-10;
        it_sm = 0;


        % Storia del miglior valore per disegno grafico
        history = zeros(1,n_it_PSO);


        %Ciclo principale 
        k=1;
        tol_v = 1e-6;
while k <= n_it_PSO 

    f_current_best = f(y_best);

        %Valutazione fitness per ogni individuo della popolazione
        for i = 1:n
            z(i) = f(x(:,i));
            % Aggiornamento pbest
            if z(i) < f(y(:,i))
                y(:,i) = x(:,i);
            end
            if f(y(:,i)) < f(y_best)
                y_best = y(:,i);
            end
        end

        % Aggiornamento contatore miglioramento
        if f(y_best) < f_current_best*(1-tol_migl)
            it_sm = 0; % reset
        else
            it_sm = it_sm + 1;
        end
        if it_sm >= n_it_sm
            flag = 2;
            break
        end
           
        
       %Aggiornamento delle posizoni e delle velocità
        for i=1:n
            r1 = rand(dim,1);
            r2 = rand(dim,1);
            v(:,i) = w*v(:,i) + c1*r1.*(y(:,i)-x(:,i)) + c2*r2.*(y_best-x(:,i));
            x(:,i) = x(:,i) + v(:,i); 
        end

        % Rispetto dei bounds
        x = max(min(x,bound), -bound);

        % Ricalcolo
        for i = 1:n
            z(i) = f(x(:,i));
        end

        % Aggiornamento miglior valore
        f_y_best = f(y_best);
        % Salvataggio storico miglior valore
        history(k) = f_y_best;


         % Aggiornamento massimo velocità
        stop_vel = max(vecnorm(v,2,1));
        if stop_vel < tol_v
            flag = 1;
            break
        end

    if dim==2
        % Aggiornamento grafico
        set(h, 'XData', x(1,:), 'YData', x(2,:), 'ZData', z);
        drawnow;


        pause(0.1);
    end

    fprintf("f_best all'iterazione %d: %d \n",k,f_y_best);

k=k+1;

end


% Taglio del vettore history in caso di break anticipato
history = history(1:k-1);


% Plot andamento miglior valore
    figure;
    plot(1:length(history), history, '-', 'LineWidth', 1.5);
    xlabel('Iterazione');
    ylabel('Miglior valore f(y_{best})');
    title(sprintf('Andamento del miglior valore PSO (f_{best} = %.6e)', f_y_best));
    grid on;


% Motivo di terminazione PSO
switch flag
    case 0
        disp("PSO terminato per numero massimo di iterazioni raggiunte");
    case 1
        disp("PSO terminato per convergenza velocità");
    case 2
        disp("PSO terminato per numero massimo di iterazioni senza miglioramento raggiunte");
end



