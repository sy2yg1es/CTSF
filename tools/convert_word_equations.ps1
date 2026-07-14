param(
    [Parameter(Mandatory = $true)]
    [string]$Path
)

$resolved = (Resolve-Path -LiteralPath $Path).Path
$word = $null
$doc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($resolved)

    $converted = 0
    foreach ($paragraph in $doc.Paragraphs) {
        $text = $paragraph.Range.Text.Trim([char]13, [char]7, ' ')
        if ($text.StartsWith('[EQ] ')) {
            $formula = $text.Substring(5)
            $range = $paragraph.Range
            $range.End = $range.End - 1
            $range.Text = $formula
            $mathRange = $paragraph.Range
            $mathRange.End = $mathRange.End - 1
            [void]$doc.OMaths.Add($mathRange)
            $paragraph.Range.OMaths.Item(1).BuildUp()
            $converted++
        }
    }

    $doc.Save()
    Write-Output "Converted equations: $converted"
}
finally {
    if ($null -ne $doc) { $doc.Close($false) }
    if ($null -ne $word) { $word.Quit() }
    if ($null -ne $doc) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($null -ne $word) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
