function Cauchy_matrix = calculate_Cauchy_matrix(x,y)

length_x = length(x);
length_y = length(y);

Cauchy_matrix = zeros(length_x,length_y);

for ix = 1:length_x
    for iy = 1:length_y
        Cauchy_matrix(ix,iy) = 1./(x(ix) - y(iy));
    end
end

end