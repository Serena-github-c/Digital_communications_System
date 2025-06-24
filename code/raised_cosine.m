function h = raised_cosine(beta, span, sps)
% Generates a Raised Cosine filter impulse response
% raised_cosine generates a raised cosine filter
% beta: roll-off factor (0 ≤ beta ≤ 1)
% span: number of symbol durations the filter spans
% sps: samples per symbol (oversampling)

    t = -span/2 : 1/sps : span/2;  % time vector in symbol durations

    h = zeros(size(t));
    for i = 1:length(t)
        if t(i) == 0
            h(i) = 1;
        elseif abs(t(i)) == 1/(2*beta)
            h(i) = (pi/4)*sinc(1/(2*beta));
        else
            h(i) = sinc(t(i)) .* cos(pi*beta*t(i)) ./ (1 - (2*beta*t(i))^2);
        end
    end

    h = h / sum(h);  % Normalize energy
end

