#!/usr/bin/env octave

fname = argv(){1}; % file name
load(fname, 'momentum');

vx = squeeze(momentum(1, :, :, :));


y = squeeze(mean(mean(vx, 1), 2));
y = y(abs(y) > 0);
y = y(4:length(y)-3)

m = length(y);
h = 0.5;
x = linspace(h*0.5, (m-0.5)*h, m);

X = [ones(m, 1) x'];

theta = (pinv(X'*X))*X'*y

