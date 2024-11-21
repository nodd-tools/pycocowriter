# Contributing Guidelines

*Pull requests, bug reports, and all other forms of contribution are welcomed and highly encouraged!* 

### Contents

- [Code of Conduct](#code-of-conduct)
- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Code Review](#code-review)
- [Coding Style](#coding-style)
- [Documentation](#documentation)
- [Certificate of Origin](#certificate-of-origin)

> **This guide serves to set clear expectations for everyone involved with the project so that we can improve it together while also creating a welcoming space for everyone to participate. Following these guidelines will help ensure a positive experience for contributors and maintainers.**

##  Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md). It is in effect at all times. We expect it to be honored by everyone who contributes to this project. Acting like an asshole will not be tolerated.

###  Bug Reports

Please include a minimal reproducible example with your bug report.

##  Feature Requests

Feature requests are welcome if they fit within the scope of the project.

Feature requests that you are willing to complete are especially welcome.  

##  Submitting Pull Requests

Please submit an issue first and get community buy-in for proposed changes before doing any work.

Please submit PRs in the smallest possible non-breaking chunks.

##  Code Review

Any code pulled into this repo should be reviewed by a maintainer.

Remember:

- **Review the code, not the author.** Look for and suggest improvements without disparaging or insulting the author. Provide actionable feedback and explain your reasoning.

- **You are not your code.** When your code is critiqued, questioned, or constructively criticized, remember that you are not your code. Do not take code review personally.

##  Coding Style

Follow the existing style.  We use the VSCode autopep8 linter.

## Documentation

Please include typing and docstrings for any classes.  We use [numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).  Our auto-documentation is configured to parse this style, so please follow this convention.

## Testing

Please write tests for your code.  Tests should be discoverable or runnable on a file-by-file basis.  Make sure all tests pass before submitting a pull request. 

    python -m unittest discover tests

##  Certificate of Origin

WHEN YOU SUBMIT CODE TO THIS REPOSITORY, YOU AGREE TO LICENSE YOUR CODE UNDER [THE LICENSE](LICENSE)

*Developer's Certificate of Origin 1.1*

By making a contribution to this project, I certify that:

> 1. The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
> 1. The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
> 1. The contribution was provided directly to me by some other person who certified (1), (2) or (3) and I have not modified it.
> 1. I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
