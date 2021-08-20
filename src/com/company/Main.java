package com.company;
import java.io.*;
import java.util.Vector;


public class Main {

    public static void main(String[] args) throws IOException {
        Vector<Vector<Float>> data= new Vector<>();
        Vector<Vector<Float>> testData= new Vector<>();
        int inputSize=0, hiddenSize=0, outputSize=0;

        //Start first Program:

        //Reading Training Data Input File
        Vector<String> lines= readFile("training.txt");
        //split each line to training example and store it in Vector<Float>
        for (int i=0; i<lines.size(); i++){
            String[] split= lines.get(i).split(" ");
            if(i==0){ //first line
                inputSize= Integer.parseInt(split[0]);
                hiddenSize= Integer.parseInt(split[1]);
                outputSize= Integer.parseInt(split[2]);
            }
            else if (i>1){ //skip second line
                Vector<Float>example= new Vector();
                for (int j=0; j<split.length; j++) {
                    if (!split[j].isEmpty())
                        example.add(Float.parseFloat(split[j]));
                }
                data.add(example);
            }
        }

        System.out.println("Size of Training Data = "+ data.size()+"");

        network n = new network(data,inputSize,hiddenSize,outputSize);
        n.Run();

        //start second program / test:

        //Reading Training Data Input File
        Vector<String> lines1= readFile("test.txt");
        //split each line to training example and store it in Vector<Float>
        for (int i=0; i<lines1.size(); i++){
            String[] split= lines1.get(i).split(" ");
            if (i>1){ //skip first and second line
                Vector<Float> test= new Vector<>();
                for (int j=0; j<split.length; j++) {
                    if (!split[j].isEmpty())
                        test.add( Float.parseFloat(split[j]) );
                }
                testData.add(test);
            }
        }

        System.out.println("\nSize of Test Data = "+ testData.size()+"");

        n.test(testData);

    }


    public static Vector<String> readFile(String fileName){
        String line = "";
        Vector<String> Data=new Vector();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            while ((line = br.readLine()) != null) {
                Data.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return Data;
    }
}
