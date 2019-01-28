cell = Perceptron;

Iteration = 2000;
LearningRate = 0.00949;
Inputs = [-0.329,1; -0.71,-0.2; 0.45,1.75; 0.32,0.44; -0.3,0.85;
          1.56,1.81; 1.7,2; 1.33,1.88; 1.9,1.22; 1.91,1.9];
Labels = [1,1,1,1,1,-1,-1,-1,-1,-1];

cell = cell.setInfo(Inputs, Labels, LearningRate, Iteration);
cell = cell.Learning();
%disp(cell.w1_revise);
figure(1);
    plot(Inputs(1:5, 1),Inputs(1:5, 2), 'ro');
    hold on
    plot(Inputs(6:10, 1), Inputs(6:10, 2),'bo');
    hold on
    x = linspace(-1, 6, 140);
    y = ((-x*cell.w(1))/cell.w(2))+cell.b/cell.w(2);
    plot(x, y);
    ylim([-2,5.5])
    legend('Class 1', 'Class 2', 'Hyper Plain');
    title('Distribution of Scatters');
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\Assignment1\Distribution.png');
    hold off
   
figure(2);
    plot(cell.w1_revise(1:30),'r-','LineWidth',1.3);
    hold on
    plot(cell.w2_revise(1:30), 'k-','LineWidth',1.3);
    hold on
    plot(cell.b_revise(1:30), 'b-','LineWidth',0.8);
    legend('w1', 'w2', 'b');
    title('Convergence of Parameters');
    print(gcf,'-dpng','C:\Users\Yunqing\Desktop\SEM2\fuzzy\Assignment1\Convergence.png');
    hold on
    ylim([-1 1]);
    hold off