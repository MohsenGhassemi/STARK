%%%Projection from D^pi ----> D
function D = rearrange_Dpi_D(D_pi,M,P)
D = zeros(M(1)*M(2)*M(3),P(1)*P(2)*P(3));
if M(3)*P(3) ~= 1
    for i=1:size(D,1)
        for j=1:size(D,2)
            S_i_q = floor((i-1)/(M(2)*M(3)));
            S_j_q = floor((j-1)/(P(2)*P(3)));
            
            i_q = i - S_i_q*M(2)*M(3);
            j_q = j - S_j_q*P(2)*P(3);
            
            S_i_r = floor((i_q - 1)/M(3));
            S_j_r = floor((j_q - 1)/P(3));
            
            
            i_r = i_q - S_i_r*M(3);
            j_r = j_q - S_j_r*P(3);
            
            s = S_i_q + S_j_q*M(1) + 1;
            r = S_i_r + S_j_r*M(2) +1;
            q = i_r + (j_r - 1)*M(3);
            
            D(i,j) = D_pi(q,r,s);
        end
    end
else
    for i=1:size(D,1)
        for j=1:size(D,2)
            S_i_q = floor((i-1)/M(2));
            S_j_q = floor((j-1)/P(2));
            
            i_q = i - S_i_q*M(2);
            j_q = j - S_j_q*P(2) ;
            
            r = S_i_q  + S_j_q*M(1) + 1;
            c = i_q + (j_q -1)*M(2);
            
            D(i,j) = D_pi(c,r);
        end
    end
end

