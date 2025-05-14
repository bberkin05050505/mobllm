# Set paths
$sourceRepo = "C:\Users\Berkin\Desktop\PostUG\Spring 2025\MEng\mobllm"
$flatRepo   = "C:\Users\Berkin\Desktop\PostUG\Spring 2025\mobllm"

# Get list of all local branches
cd $sourceRepo
$branches = git for-each-ref --format='%(refname:short)' refs/heads/

foreach ($branch in $branches) {
    Write-Host "`nProcessing branch: $branch" -ForegroundColor Cyan

    # Checkout the branch in source repo
    git checkout $branch

    # Copy all files except .git to the flat repo
    robocopy $sourceRepo $flatRepo /E /XD .git

    # Go to flat repo
    cd $flatRepo

    # Create and switch to the branch
    git checkout -b $branch

    # Stage and commit
    git add .
    git commit -m "Flattened snapshot of $branch"

    # Go back to source repo
    cd $sourceRepo
}

Write-Host "`n✅ All branches flattened and committed in: $flatRepo" -ForegroundColor Green
