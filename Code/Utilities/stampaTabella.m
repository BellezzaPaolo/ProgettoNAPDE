% small script to print the test tables in a confortable way to import it in
% the Latex report

clc
format short
disp('textbf{$ \')
fprintf('eta$} & Stochastic ')
for i=1:length(n_fine)
    fprintf('& $n_{fine}=%d$ ',n_fine(i))
end
disp('\\')
disp('\hline')
for j=1:length(learning_rate)
    fprintf('     $%.1e$ & ',learning_rate(j))
    fprintf('$%.1e$ ',M_full(1,j,1))
    for i= 2:length(n_fine)+1
        fprintf('& $%.1e$ ',M_full(1,j,i))
    end
    disp('\\')
    disp('   \hline')
end