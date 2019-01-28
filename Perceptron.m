classdef Perceptron
    
    properties % define the variables involved
        Inputs;
        Labels;
        LearningRate;
        Iteration;
        w;
        b;
        w1_revise;
        w2_revise;
        b_revise;
        loss;
    end
    
    methods  % define the functions used
        function obj = Perceptron() % constructor
        end
        
        function obj = setInfo(obj,Inputs, Labels, LearningRate, Iteration)
            obj.Inputs = Inputs;
            obj.Labels = Labels;
            obj.LearningRate = LearningRate;
            obj.Iteration = Iteration;
        end
        
        function obj = Learning(obj)
            [m, n] = size(obj.Inputs);
            obj.w = rand(1, n);
            obj.b = 0.8;
            obj.w1_revise = zeros(1, obj.Iteration);
            obj.w2_revise = zeros(1, obj.Iteration);
            obj.b_revise = zeros(1,- obj.Iteration);
            for N=1:obj.Iteration
                for set= 1:m
                    output = sign(obj.Inputs(set,:)*(obj.w)' - obj.b );
                    error = obj.Labels(set) - output;
                    obj.w = obj.w + obj.LearningRate * error * obj.Inputs(set);
                    obj.b = obj.b - obj.LearningRate * error;
                end
            obj.w1_revise(1, N) = obj.w(1);
            obj.w2_revise(1, N) = obj.w(2);
            obj.b_revise(1, N) = obj.b;    
            end
        end
        
        function obj = Prediction(obj)
            obj.value = sign( obj.Inputs(1,:)*(obj.w)' + obj.b);
        end
    end  
end