% Initial conditions are 2000 ants, 70% of which are active
activeForagers = 1400;
inactiveForagers = 600;
% a is the density of active foragers
%a = activeForagers / (activeForagers + inactiveForagers);
mu = 0.679592;
%k = 62.6;
%k = 5;
k = (6 + 1.77148446) / 2;
%k = 1.77148446;
beta = 0.689091;

tspan = [0 10]
a0 = 0.7

[t, y] = ode45(@(t, a) -(mu * a) + (beta * k * a) - (beta * k * a^2), tspan, a0);

plot(t, y, '-o')

%rate = -(mu * a) + (beta * k * a) - (beta * k * a^2)

%syms a(t)

%ode = diff(a, t) == -(mu * a) + (beta * k * a) - (beta * k * a^2)
%ode = diff(a, t) == (43.1370966*a) - (43.1370966*a^2) - (0.679592*a)
%ySol(t) = dsolve(ode)

%cond = a(0) == 0.7;

%ySol(t) = dsolve(ode, cond)

%ySol(7)

%-(mu * a) + (beta * k * a) - (beta * k * a * a)
%expr = -(mu * x) + (beta * k * x) - (beta * k * x^2)

%in

