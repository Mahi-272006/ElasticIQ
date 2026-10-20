from src.data.loader import inspect_csv, list_raw_files


def main():
    print("ELASTICIQ - DATA FILE CHECK")
    print("=" * 40)

    for file_info in list_raw_files():
        if file_info["exists"]:
            status = "FOUND"
        else:
            status = "MISSING"

        print(
            "{}: {} | {} MB".format(
                file_info["filename"],
                status,
                file_info["size_mb"],
            )
        )

    print("\nCALENDAR FILE INSPECTION")
    print("=" * 40)

    calendar = inspect_csv("calendar.csv")

    print("Columns:", calendar["columns"])
    print("\nFirst five rows:")
    print(calendar["sample"].to_string(index=False))


if __name__ == "__main__":
    main()
