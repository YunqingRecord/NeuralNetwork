cell = Perceptron; % acquire class

Iteration = 2000;
LearningRate = 0.00549;
Inputs = [-0.8,-0.755; -0.781,0.33; -0.63,-0.24; -0.16,-0.44; -0.11,0.55;
          1.36,0.41; 1.59,1.02; 1.43,0.08; 1.19,1.22; 1,-0.59;
          0.729,0.233; 0.129,0.33; 0.335,-0.015; 0.332,0.74; 0.55,0.9;
          0.06,1.1; 0.7,1.42; 0.93,1.24; 0.57,-0.77; 0.01,1.3]; % augmented Datasets
Labels = [1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];

cell = cell.setInfo(Inputs, Labels, LearningRate, Iteration);
cell = cell.Learning();
%disp(cell.w1_revise);
figure(1);
    [XXX,YYY]=meshgrid(-1:0.05:2);
    RRR=zeros(size(XXX));

    for x = 1:61
        for y = 1:61
            % Obtain an input data vector
            I = [ XXX(x,y) YYY(x,y) ];
            % Calculate the output
            tmpout = I(1)*cell.w(1) + I(2)*cell.w(2) + cell.b;
            if (tmpout > 0)
                out = 1; RRR(x,y) = 1;
            else
                out = -1; RRR(x,y) = -1;
            end
        end
    end

    Z1 = [1,1,1,1,1,1,1,1,1,1];
    Z2 = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
    % figure(1);
    plot3(Inputs(1:10, 1),Inputs(1:10, 2),Z1, 'ro');
    hold on
    plot3(Inputs(11:20, 1),Inputs(11:20, 2),Z1, 'o'); % for convinence, set the points upside 3-d and view(2)
    %legend('Class 1','Class 2');
    title('Partition of 2-D Plane');
    hold on
    %mesh(XXX,YYY,RRR),view(2),colorbar; 
    %hold on
    surf(XXX,YYY,RRR),view(2),colorbar ;
    shading interp
    hold off
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\Assignment1\Plot2.png');

   
figure(2);
    plot(cell.w1_revise(1:80),'r-','LineWidth',1.3);
    hold on
    plot(cell.w2_revise(1:80), 'k-','LineWidth',1.3);
    hold on
    plot(cell.b_revise(1:80), 'b-','LineWidth',0.8);
    hold on 
    legend('w1', 'w2', 'b');
    title('Convergence of Parameters');
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\Assignment1\NonConvergence1.png');
    hold on
    %ylim([-5 10]);
    hold off
figure(3);
    plot(cell.error_cal(1:40), '-','LineWidth',0.8);
    
