package main.java.experiments;

import cf4j.Kernel;
import io.jenetics.*;
import io.jenetics.engine.Codec;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.ext.SingleNodeCrossover;
import io.jenetics.prog.ProgramChromosome;
import io.jenetics.prog.ProgramGene;
import io.jenetics.prog.op.Const;
import io.jenetics.prog.op.Op;
import io.jenetics.prog.op.Var;
import io.jenetics.util.ISeq;
import main.java.mf.Emf;
import main.java.qualityMeasures.QualityMeasures;
import org.apache.commons.cli.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

public class GeneticProgramingOptimization {

	private static final String BINARY_FILE = "datasets/ml100k.cf4j";

	private static int NUM_TOPICS = 6;
	private static double REGULARIZATION = 0.055;
	private static double LEARNING_RATE = 0.0001;
	private static int GENS = 150;
	private static double PBMUT = 0.1;
	private static double PBX = 0.5;
	private static int POP_SIZE = 50;

//	private static final String BINARY_FILE = "datasets/filmtrust.cf4j";
//
//	private static double REGULARIZATION = 0.095;
//	private static double LEARNING_RATE = 0.0035;
//	private static int GENS = 150;
//	private static double PBMUT = 0.1;
//	private static double PBX = 0.5;
//	private static int POP_SIZE = 50;


	private static int NUM_ITERS = 100;


	// Jenetics operations
	private final static Op<Double> sin = Op.of("sin", 1, v -> Math.sin(v[0]));
	private final static Op<Double> cos = Op.of("cos", 1, v -> Math.cos(v[0]));
	private final static Op<Double> atan = Op.of("atan", 1, v -> Math.atan(v[0]));
	private final static Op<Double> exp = Op.of("exp", 1, v -> Math.exp(v[0]));
	private final static Op<Double> log = Op.of("log", 1, v -> Math.log(v[0]));
	private final static Op<Double> inv = Op.of("inv", 1, v -> 1.0 / v[0]);
	private final static Op<Double> sign = Op.of("--", 1, v -> -v[0]);
	private final static Op<Double> add = Op.of("+", 2, v -> v[0] + v[1]);
	private final static Op<Double> sub = Op.of("-", 2, v -> v[0] - v[1]);
	private final static Op<Double> times = Op.of("*", 2, v -> v[0] * v[1]);
	private final static Op<Double> pow = Op.of("pow", 2, v -> Math.pow(v[0], v[1]));


	// Jenetics terminals
	private final static Const<Double> zero = Const.of("Zero", 0.0);
	private final static Const<Double> one = Const.of("One", 1.0);
	private static PrintWriter output;


	public static void main (String [] args) {
		CommandLineParser parser = new DefaultParser();
		Options options = new Options();
		options.addOption(new Option( "help", "print this message" ));
		options.addOption(
				Option.builder("lambda")
				.longOpt("lambda")
				.desc(String.format("default: %.6f", REGULARIZATION))
				.hasArg()
				.argName("VALUE")
				.type(double.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("gamma")
				.longOpt("gamma")
				.desc(String.format("default: %.6f", LEARNING_RATE))
				.hasArg()
				.argName("VALUE")
				.type(double.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("iters")
				.longOpt("iters")
				.desc("default: " + NUM_ITERS)
				.hasArg()
				.argName("VALUE")
				.type(int.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("generations")
				.longOpt("generations")
				.desc("Number of generations, default: " + GENS)
				.hasArg()
				.argName("N")
				.type(int.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("K")
				.desc("Number of topics, default: " + NUM_TOPICS)
				.hasArg()
				.argName("K")
				.type(int.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("pop")
				.desc("Population size, default: " + POP_SIZE)
				.hasArg()
				.longOpt("population-size")
				.argName("pop")
				.type(int.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("pbx")
				.desc("Crossover probability, default: " + PBX)
				.hasArg()
				.longOpt("crossover-prob")
				.argName("pbx")
				.type(double.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("pbmut")
				.desc("Mutation probability, default: " + PBMUT)
				.hasArg()
				.longOpt("mutation-prob")
				.argName("pbmut")
				.type(double.class)
				.valueSeparator()
				.build());

		try {
			CommandLine line = parser.parse(options, args);
			if (line.hasOption("help")){
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("gmf", options);
				return;
			}

			if (line.hasOption("lambda")){
				REGULARIZATION = Double.parseDouble(line.getOptionValue("lambda"));
			}
			if (line.hasOption("gamma")){
				LEARNING_RATE = Double.parseDouble(line.getOptionValue("gamma"));
			}
			if (line.hasOption("iters")){
				NUM_ITERS = Integer.parseInt(line.getOptionValue("iters"));
			}
			if (line.hasOption("generations")){
				GENS = Integer.parseInt(line.getOptionValue("generations"));
			}
			if (line.hasOption("K")) {
				NUM_TOPICS = Integer.parseInt(line.getOptionValue("K"));
			}
			if (line.hasOption("pop")) {
				POP_SIZE = Integer.parseInt(line.getOptionValue("pop"));
			}
			if (line.hasOption("pbx")) {
				PBX = Double.parseDouble(line.getOptionValue("pbx"));
			}
			if (line.hasOption("pbmut")) {
				PBMUT = Double.parseDouble(line.getOptionValue("pbmut"));
			}
		} catch (ParseException e) {
			System.out.println( "Unexpected exception:" + e.getMessage() );
		}

		Kernel.getInstance().readKernel(BINARY_FILE);

		final ISeq<Op<Double>> operations = ISeq.of(sin, cos, atan, exp, log, inv, sign, add, sub, times, pow);

		ISeq<Op<Double>> inputs = ISeq.empty();
		// Terminal nodes: variables
		for (int i = 0; i < NUM_TOPICS; i++) {
			inputs = inputs.append(
					Var.of("pu"+i, i),
					Var.of("qi"+i, i + NUM_TOPICS)
			);
		}
		inputs = inputs.append(zero, one);

		// Tree building
		final Codec<ProgramGene<Double>, ProgramGene<Double>> codec = Codec.of(
				Genotype.of(ProgramChromosome.of(
						6,
						ch -> ch.getRoot().size() <= 150,
						operations,
						inputs
				)), Genotype::getGene
		);

		final Engine<ProgramGene<Double>, Double> engine = Engine
				.builder(GeneticProgramingOptimization::fitness, codec)
				.minimizing()
				.offspringSelector(new TournamentSelector<>())
				.alterers(
						new SingleNodeCrossover<>(PBX),
						new Mutator<>(PBMUT))
				.survivorsSelector(new EliteSelector<ProgramGene<Double>, Double>(2, new MonteCarloSelector<>()))
				.populationSize(POP_SIZE)
				.build();

		// Output file with unique filename
		SimpleDateFormat df = new SimpleDateFormat("ddMMyy-hhmmss.SSS");
		try {
			Date d = new Date();
			File outputFile = new File(df.format(d) + ".csv");
			output = new PrintWriter(outputFile);
		} catch (IOException e) {
			e.printStackTrace();
		}

		final EvolutionResult<ProgramGene<Double>, Double> population = engine.stream()
				.limit(GENS)
				.peek(GeneticProgramingOptimization::update)
				.peek(GeneticProgramingOptimization::toFile)
				.collect(EvolutionResult.toBestEvolutionResult());

//		try {
//			if (popOutput.canWrite())
//				IO.object.write(population.getPopulation(), popOutput);
//			output.flush();
//			output.close();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

		output.close();

		System.out.println(population.getBestPhenotype().getGenotype().getGene().toParenthesesString());
	}

	private static double fitness(final ProgramGene<Double> program) {
		String func = program.toParenthesesString()
				.replace("("," ")
				.replace(")"," ")
				.replace(",", " ");

		Emf emf = new Emf(func, NUM_TOPICS, NUM_ITERS, REGULARIZATION, LEARNING_RATE);

		double error = QualityMeasures.MAE(emf);

		return Double.isNaN(error) ? 4.0 : error;
	}

	private static void update(final EvolutionResult<ProgramGene<Double>, Double> result) {
		String info = String.format(
				"%d/%d:\tbest=%.4f\tinvalids=%d\tavg=%.4f\tbest=%s",
				result.getGeneration(),
				GENS,
				result.getBestFitness(),
				result.getInvalidCount(),
				result.getPopulation().stream().collect(Collectors.averagingDouble(Phenotype::getFitness)),
				result.getBestPhenotype().getGenotype().getGene().toParenthesesString());
		System.out.println(info);
	}

	private static void toFile(final EvolutionResult<ProgramGene<Double>, Double> result) {
		double[] fitnesses = result.getPopulation().stream().mapToDouble(Phenotype::getFitness).toArray();
		List<CharSequence> fs = new ArrayList<>();

		for (double fitness : fitnesses) {
			fs.add(String.valueOf(fitness));
		}

		output.println(result.getGeneration() + "," + String.join(",", fs));

	}
}
