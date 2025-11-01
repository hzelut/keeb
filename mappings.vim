"
" Mapping
"
let mapleader=','
function! Mapmap(lhs, rhs)
  execute 'map '.a:lhs.' '.a:rhs
  execute 'imap '.a:lhs.' <Esc>'.a:rhs
endfunction
" Cursor
noremap l h
noremap L b
noremap i j
noremap I <C-d>
noremap s k
noremap S <C-u>
noremap t l
noremap T w
" Mode
noremap e i
noremap E I
noremap <C-r> R
" Operation
noremap r c
noremap R C
noremap c y
noremap C Y
" Etc
noremap f za
noremap F zR
noremap m @
noremap M q
noremap j :join!<CR>
noremap J :.-1join!<CR>
noremap y "
noremap Y :registers<CR>
noremap U <C-r>
noremap z :call CycleList('l', 'n')<CR>
noremap Z :call CycleList('l', 'p')<CR>
noremap b g;
noremap B g,
noremap <Space> o<Esc>
noremap <BackSpace> X
noremap ( [(
noremap ) ])
noremap { [{
noremap } ]}
noremap ; %
noremap + <C-a>
noremap - <C-x>
noremap h <Plug>(YCMHover)
noremap H :noh<CR>
noremap w :w<CR>

" Cursor
call Mapmap('<Leader><Left>',  '<C-w>h')
call Mapmap('<Leader><Right>', '<C-w>l')
call Mapmap('<Leader><Up>',    '<C-w>k')
call Mapmap('<Leader><Down>',  '<C-w>j')
call Mapmap('<PageUp>',        '<C-b>')
call Mapmap('<PageDown>',      '<C-f>')
" Window
call Mapmap('<Leader>w', ':wqall<CR>')
call Mapmap('<Leader>C', '<C-w>c')
call Mapmap('<Leader>v', ':vsplit<CR>')
call Mapmap('<Leader>h', ':split<CR>')
" Buffer
call Mapmap('<Leader>c', ':call CloseBuf()<CR>')
call Mapmap('<Leader>e', ':bprevious<CR>')
call Mapmap('<Leader>a', ':bnext<CR>')
" Sizing
call Mapmap('<Leader>+', '<C-w>5+')
call Mapmap('<Leader>-', '<C-w>5-')
call Mapmap('<Leader><', '<C-w>5<')
call Mapmap('<Leader>>', '<C-w>5>')
call Mapmap('<Leader>=', '<C-w>=')
call Mapmap('<Leader>_', '<C-w>_')
call Mapmap('<Leader>\|', '<C-w>\|')
" List
call Mapmap('<Leader>i', ':call ToggleList("l")<CR>')
call Mapmap('<Leader>s', ':call ToggleList("c")<CR>')
call Mapmap('<Leader>z', ':call CycleList("c","n")<CR>')
call Mapmap('<Leader>Z', ':call CycleList("c","p")<CR>')
" YCM
call Mapmap('<Leader>p', ':YcmCompleter GoToAlternateFile<CR>')
call Mapmap('<Leader>r', ':YcmCompleter GoToReferences<CR>:call AfterYcm()<CR>')
call Mapmap('<Leader>g', ':YcmCompleter GoTo<CR>')
call Mapmap('<Leader>f', '<Plug>(YCMFindSymbolInWorkspace)')
" Nerdtree
call Mapmap('<Leader>l', ':NERDTreeToggle<CR>')
let g:NERDTreeMenuUp='s'
let g:NERDTreeMenuDown='i'
let g:NERDTreeMapRefresh='l'
let g:NERDTreeMapCustomOpen='t'
" Etc
call Mapmap('<Leader>b', '<C-o>')
call Mapmap('<Leader>B', '<C-i>')
