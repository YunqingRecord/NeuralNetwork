plant = NN;
Inputs = [-0.8,-0.755; -0.781,0.33; -0.63,-0.24; -0.16,-0.44; -0.11,0.55;
          1.36,0.41; 1.59,1.02; 1.43,0.08; 1.19,1.22; 1,-0.59;
          0.729,0.233; 0.129,0.33; 0.335,-0.015; 0.332,0.74; 0.55,0.9;
          0.06,1.1; 0.7,1.42; 0.93,1.24; 0.57,-0.77; 0.01,1.3];
Labels = [1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
Inputs = Inputs';
Iteration = 2000;
LearningRate = 0;
plant = plant.setInfo(Inputs, Labels, LearningRate, Iteration); % insert info.
plant = plant.Learning();
% Test NN and Display results
[XXX,YYY]=meshgrid(-1:0.05:2);
RRR=zeros(size(XXX));

for x = 1:61
    for y = 1:61
        % Obtain an input data vector
        I = [ XXX(x,y) YYY(x,y) ];
        % Calculate the output
        tmpout = plant.net(I');
        if (tmpout > 0)
            out = 1; RRR(x,y) = 1;
        else
            out = -1; RRR(x,y) = -1;
        end
    end
end

Z1 = [1,1,1,1,1,1,1,1,1,1];
Z2 = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
figure(1);

plot3(Inputs(1, 1:10),Inputs(2, 1:10),Z1, 'ro');
hold on
plot3(Inputs(1, 11:20),Inputs(2, 11:20),Z2, 'go');
legend('Class 1','Class 2');
title('Partition of 2-D Plane');
hold on
%mesh(XXX,YYY,RRR),view(2),colorbar; 
%hold on
surf(XXX,YYY,RRR),view(2),colorbar ;
shading interp
hold on
