from alchemind.generation import GeneticGenerator, GenerationConfig, crossover, mutate
from alchemind.utils.chem import is_valid_smiles

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
BENZENE = "c1ccccc1"


def test_mutate_returns_valid_distinct_molecules():
    muts = mutate(ASPIRIN, n=5)
    assert len(muts) >= 1
    assert all(is_valid_smiles(m) for m in muts)
    assert ASPIRIN not in muts  # mutations differ from parent


def test_mutate_invalid_input_returns_empty():
    assert mutate("not_a_molecule") == []


def test_crossover_produces_valid_children():
    children = crossover(ASPIRIN, "CC(=O)Nc1ccc(O)cc1", n=5)
    assert all(is_valid_smiles(c) for c in children)


def test_genetic_generator_optimizes_fitness():
    # Fitness = number of heavy atoms -> generator should grow molecules.
    from alchemind.utils.chem import mol_from_smiles

    def fitness(smi):
        m = mol_from_smiles(smi)
        return m.GetNumHeavyAtoms() if m else -1

    gen = GeneticGenerator(GenerationConfig(population_size=25, generations=4, seed=1))
    results = gen.generate([BENZENE], fitness, n=10)
    assert len(results) >= 1
    smiles, scores = zip(*results)
    assert all(is_valid_smiles(s) for s in smiles)
    # Best evolved molecule should be at least as large as the seed (6 atoms).
    assert max(scores) >= 6
