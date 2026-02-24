%Implementazione funzione obiettivo 

function y = f(x)

    %Funzione Rastrigin
    d=length(x);
    y=10*d+sum(x.^2-10*cos(2*pi*x));

    
