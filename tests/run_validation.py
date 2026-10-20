from src.data.validation import run_sample_validation


def main():
    reports = run_sample_validation()

    print("\nELASTICIQ - DATA VALIDATION REPORT")
    print("=" * 50)

    for report in reports:
        print(f"\nDataset: {report['file']}")
        print(f"Rows inspected: {report['sample_rows']}")

        for result in report["results"]:
            status = "PASS" if result["passed"] else "FAIL"

            print(f"  [{status}] {result['check']}")

            if result.get("missing_columns"):
                print(
                    "         Missing:",
                    result["missing_columns"],
                )

            if "day_column_count" in result:
                print(
                    "         Day columns:",
                    result["day_column_count"],
                )

    print("\nValidation run complete.")


if __name__ == "__main__":
    main()
