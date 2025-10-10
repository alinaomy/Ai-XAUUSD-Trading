# Contributing to AI-XAUUSD Trading System

Thank you for your interest in contributing to the AI-XAUUSD Trading System! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## 🤝 Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of reinforcement learning and trading concepts

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/AI-XAUUSD-Trading.git
   cd AI-XAUUSD-Trading
   ```

3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/JonusNattapong/AI-XAUUSD-Trading.git
   ```

## 🛠️ Development Setup

### Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:
```bash
pip install pre-commit
pre-commit install
```

## 📝 Contributing Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write descriptive variable and function names
- Add docstrings to all functions and classes
- Keep line length under 88 characters (Black default)

### Commit Messages

Use clear, descriptive commit messages following this format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing related changes
- `chore`: Maintenance tasks

Examples:
```
feat(trading): add confidence-based position sizing
fix(env): resolve memory leak in trading environment
docs(readme): update installation instructions
```

### Branch Naming

Use descriptive branch names:
- `feature/description-of-feature`
- `bugfix/issue-description`
- `docs/update-documentation`
- `refactor/component-name`

## 🧪 Testing

### Running Tests

Run the full test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=ai_xauusd_trading --cov-report=html
```

Run specific test files:
```bash
pytest tests/test_trading_env.py
```

### Writing Tests

- Write unit tests for all new functions
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies
- Aim for >80% code coverage

Example test structure:
```python
import pytest
from ai_xauusd_trading.trading_env import TradingEnvironment

class TestTradingEnvironment:
    def test_initialization(self):
        env = TradingEnvironment()
        assert env.capital == 1000

    def test_step_execution(self):
        env = TradingEnvironment()
        state, reward, done, info = env.step(1.0)
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
```

## 📚 Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Document parameters, return values, and exceptions

Example:
```python
def calculate_position_size(self, confidence: float, risk_per_trade: float) -> float:
    """Calculate position size based on confidence and risk parameters.

    Args:
        confidence: Model confidence score (0.0 to 1.0)
        risk_per_trade: Maximum risk per trade as decimal

    Returns:
        Position size as percentage of capital

    Raises:
        ValueError: If confidence is outside valid range
    """
```

### Documentation Updates

- Update README.md for new features
- Update docstrings when changing function signatures
- Add examples for new functionality
- Update installation and usage instructions

## 🔄 Pull Request Process

1. **Create a Branch**: Create a feature branch from `main`
2. **Make Changes**: Implement your changes with tests
3. **Run Tests**: Ensure all tests pass
4. **Update Documentation**: Update relevant docs
5. **Commit Changes**: Use clear commit messages
6. **Push Branch**: Push your branch to GitHub
7. **Create PR**: Open a pull request with a clear description

### PR Template

Use this template for pull requests:

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests pass
- [ ] No breaking changes
```

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Clear Title**: Describe the issue concisely
2. **Environment**: Python version, OS, dependencies
3. **Steps to Reproduce**: Minimal steps to reproduce the issue
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens
6. **Error Messages**: Include full error output
7. **Code Sample**: Minimal code to reproduce the issue

### Feature Requests

For feature requests, please include:

1. **Clear Title**: Describe the desired feature
2. **Problem Statement**: What problem does this solve?
3. **Proposed Solution**: How should it work?
4. **Alternatives**: Other solutions considered
5. **Additional Context**: Screenshots, examples, etc.

## 🎯 Areas for Contribution

### High Priority
- [ ] Performance optimizations
- [ ] Additional technical indicators
- [ ] More sophisticated risk management
- [ ] Live trading integration improvements

### Medium Priority
- [ ] Web dashboard for monitoring
- [ ] Backtesting framework improvements
- [ ] Additional market regime types
- [ ] Model interpretability features

### Low Priority
- [ ] Mobile app interface
- [ ] Alternative data sources
- [ ] Multi-asset trading support
- [ ] Social trading features

## 📞 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs and request features
- **Discord**: Join our community Discord (link in README)

## 🙏 Recognition

Contributors will be recognized in:
- README.md acknowledgments section
- CHANGELOG.md for significant contributions
- GitHub repository contributors list

Thank you for contributing to the AI-XAUUSD Trading System! 🚀