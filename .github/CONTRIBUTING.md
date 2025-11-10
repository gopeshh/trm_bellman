# Contributing Guidelines

## Branch Naming Conventions

Use the following prefixes for branch names:

- `feat/` - New features or capabilities
  - Example: `feat/rl-upi-scaffold`
  - Example: `feat/rl-edit-policy`
  - Example: `feat/rl-value-head`
  - Example: `feat/rl-training-loop`

- `perf/` - Performance improvements
  - Example: `perf/spectral-contraction`

- `exp/` - Experimental work or ablations
  - Example: `exp/ablation-layer-depth`

- `fix/` - Bug fixes
  - Example: `fix/loss-computation`

- `docs/` - Documentation updates
  - Example: `docs/setup-guide`

- `test/` - Test additions or improvements
  - Example: `test/unit-coverage`

## Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `refactor`: Code refactoring
- `style`: Code style changes (formatting, etc.)
- `chore`: Maintenance tasks

### Examples

```bash
feat(rl): add value head with EMA target

test(rl): unit tests for K-step returns

docs(rl): README quickstart

perf(transformer): optimize attention computation

fix(dataset): handle edge case in maze generation
```

## Pull Request Process

1. Create a feature branch following the naming conventions above
2. Make your changes with conventional commits
3. Push your branch to the repository
4. Open a pull request using the provided template
5. Ensure all sections of the PR template are filled out:
   - Summary (what/why)
   - Design (files, APIs)
   - Risk & Mitigation
   - Test Plan (commands + expected logs)
   - Metrics/Charts (W&B runs/IDs if applicable)
   - Scope Creep (out-of-scope items)
6. Request review from appropriate team members
7. Address feedback and update the PR as needed

## Issue Labels

Use appropriate labels when creating issues:

### Area Labels
- `area:rl` - Reinforcement learning related work
- `area:training` - Training infrastructure and optimization
- `area:models` - Model architecture changes
- `area:dataset` - Dataset generation or processing

### Task Labels
- `task:sudoku` - Sudoku puzzle solving
- `task:maze` - Maze navigation
- `task:arc` - ARC (Abstraction and Reasoning Corpus) challenges

### Priority Labels
- `priority:p0` - Critical, blocking issues
- `priority:p1` - High priority
- `priority:p2` - Medium priority
- `priority:p3` - Low priority, nice to have

### Type Labels
- `type:bug` - Bug reports
- `type:feature` - Feature requests
- `type:docs` - Documentation improvements
- `type:question` - Questions or discussions

## Branch Protection

The `main` branch is protected with the following requirements:

- **Status checks must pass** before merging
  - Unit tests must pass
  - Linting checks must pass (if configured)
- **Pull request reviews** are required
- **Up-to-date branches** - branches must be up to date with `main` before merging
- **No direct pushes** to `main` - all changes must go through pull requests

To ensure your PR can be merged:

```bash
# Run tests locally before pushing
python -m pytest tests/

# Keep your branch updated with main
git fetch upstream
git rebase upstream/main
```

## Code Quality

- Ensure code follows existing style patterns in the repository
- Add tests for new functionality
- Update documentation as needed
- Run linters and formatters before committing
- Verify all unit tests pass before creating a PR
