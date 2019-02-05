cell = Perceptron; % acquire class

Iteration = 2000;
LearningRate = 0.55;
Inputs = [-0.329,1; -0.71,-0.2; 0,0.75; 0.12,-0.24; -0.3,0.85;
          0.81,0.61; 0.72,0.9; 1.03,0.48; 1.27,0.42; 1.41,1.2]; % Datasets
Labels = [1,1,1,1,1,-1,-1,-1,-1,-1];

cell = cell.setInfo(Inputs, Labels, LearningRate, Iteration);
cell = cell.Learning();
figure(1);
    [XXX,YYY]=meshgrid(-1:0.05:2);
    RRR=zeros(size(XXX));

    for x = 1:61
        for y = 1:61
            % Obtain an input data vector / binary feature
            I = [ XXX(x,y) YYY(x,y) ];
            % Calculate the output axis z
            tmpout = I(1)*cell.w(1) + I(2)*cell.w(2) + cell.b;
            if (tmpout > 0) % default threshold
                out = 1; RRR(x,y) = 1;
            else
                out = -1; RRR(x,y) = -1;
            end
        end
    end

    Z1 = [1,1,1,1,1];
    Z2 = [-1,-1,-1,-1,-1];
    % figure(1);
    plot3(Inputs(1:5, 1),Inputs(1:5, 2),Z1, 'ro');
    hold on
    plot3(Inputs(6:10, 1),Inputs(6:10, 2),Z1, 'go'); % for convinence, set the points upside 3-d and view(2)
    %legend('Class 1','Class 2');
    title('Partition of 2-D Plane');
    hold on
    %mesh(XXX,YYY,RRR),view(2),colorbar; 
    %hold on
    surf(XXX,YYY,RRR),view(2),colorbar ;
    shading interp
    hold off
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\Assignment1\Plot1.png');

   
figure(2); % show the convergence of the partameters of Perceptron
    plot(cell.w1_revise(1:40), 'r-','LineWidth',1.3); % w1
    hold on
    plot(cell.w2_revise(1:40), 'k-','LineWidth',1.3);% w2
    hold on
    plot(cell.b_revise(1:40), 'b-','LineWidth',0.8); % b
    hold on 
    plot(cell.error_cal(1:40), '-','LineWidth',0.8); % total error
    legend('w1', 'w2', 'b', 'error'); % cutline
    title('Convergence of Parameters');
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\Assignment1\Linear_Convergence.png');
    hold on
    ylim([-5 5]);
    hold off