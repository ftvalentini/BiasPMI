#!/bin/bash

iconv -c -f utf8 -t ascii//TRANSLIT $1 | # removes non ascii characters
  tr '[:upper:]' '[:lower:]' | # replace caps with low
  sed "s/[^a-z0-9]*[ \t\n\r][^a-z0-9]*/ /g" | # remove non-alphanumeric between whitespace
  sed "s/[^a-z0-9]*$//g" | # remove non-alphanumeric at the end
  sed "s/^[^a-z0-9]*//g" | # remove non-alphanumeric at beginning
  sed 's/[][]//g' | # remove square brackets
  sed 's/[[:punct:]]"/ /g' | # replace punct followed by quotation
  sed 's/[>|]/ /g' | # replace pipes
  sed 's/"/ /g' | # replace quotations
  sed 's/^ *//; s/ *$//; /^$/d' # remove empty lines and extra whitespace
